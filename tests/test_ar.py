"""The autoregressive baseline: causality, weight tying, PAD handling, probing.

The load-bearing test here is the first one. Everything else about the AR arm is
bookkeeping that a shape assertion catches; a causal mask that leaks is a silent
defect that makes the baseline win for the wrong reason, and no loss curve would
look wrong.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from ehrjepa.data.tokenize import PAD_ID
from ehrjepa.eval import probe
from ehrjepa.models import EHRAR, EHRJEPAConfig
from ehrjepa.models.encoder import Encoder
from ehrjepa.objectives.ar import ar_loss, next_code_targets
from ehrjepa.train.config import load_config
from ehrjepa.train.pretrain import Trainer

REPO = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = REPO / "configs" / "pretrain_debug.yaml"
DEMO_CACHE = REPO / "data" / "cache" / "mimic-demo"

requires_cache = pytest.mark.skipif(
    not (DEMO_CACHE / "meta.json").exists(), reason="mimic-demo cache is not built"
)

VOCAB = 64


def _config(**overrides) -> EHRJEPAConfig:
    values = dict(
        vocab_size=VOCAB,
        dim=32,
        depth=3,
        heads=4,
        pred_dim=16,
        pred_depth=1,
        pred_heads=2,
        dropout=0.0,
        n_freq=8,
        causal=True,
    )
    values.update(overrides)
    return EHRJEPAConfig(**values)


def _batch(batch: int = 3, length: int = 24, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    attention = torch.ones(batch, length, dtype=torch.long)
    attention[-1, length - 5 :] = 0
    return {
        "code_id": torch.randint(1, VOCAB, (batch, length), generator=g),
        "value_bin": torch.randint(0, 11, (batch, length), generator=g),
        "value_z": torch.randn(batch, length, generator=g),
        "age": torch.rand(batch, length, generator=g) * 90,
        "log_delta": torch.rand(batch, length, generator=g) * 8,
        "attention_mask": attention,
    }


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #


def test_causal_encoder_output_at_i_ignores_every_event_after_i() -> None:
    """Perturb event ``j`` and only outputs at positions ``>= j`` may move.

    Run over every ``j``, not a single spot check: an off-by-one in the triangle
    (``tril`` vs ``tril(-1)``, or a CLS column that shifts the diagonal) shows up
    at exactly one ``j`` and would survive a test that probed the middle.
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(7)
    length, dim = 12, 16
    encoder = Encoder(dim, depth=3, heads=4, mlp="gelu", causal=True).eval()
    tokens = torch.randn(1, length, dim)
    valid = torch.ones(1, length, dtype=torch.bool)
    with torch.no_grad():
        base = encoder(tokens, valid).tokens
        for j in range(length):
            perturbed = tokens.clone()
            # Random, not a constant: pre-LN strips a uniform shift, so adding
            # ``+10`` to every dimension of a token is invisible by construction
            # and would make this test pass on a leaky mask too.
            perturbed[0, j] += torch.randn(dim, generator=g) * 10.0
            after = encoder(perturbed, valid).tokens
            moved = (after - base).abs().amax(dim=-1)[0]
            assert torch.allclose(moved[:j], torch.zeros(j), atol=1e-6), f"leak before j={j}"
            assert moved[j] > 1e-4, f"position {j} did not react to its own perturbation"


def test_bidirectional_encoder_does_leak_the_future() -> None:
    """The control for the test above: without ``causal`` the property is false."""
    torch.manual_seed(0)
    dim = 16
    encoder = Encoder(dim, depth=2, heads=4, mlp="gelu", causal=False).eval()
    tokens = torch.randn(1, 8, dim)
    valid = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        base = encoder(tokens, valid).tokens
        perturbed = tokens.clone()
        perturbed[0, 7] += torch.randn(dim) * 10.0
        moved = (encoder(perturbed, valid).tokens[0, :7] - base[0, :7]).abs().amax()
    assert moved > 1e-4


def test_causal_cls_row_is_a_constant_the_sequence_cannot_reach() -> None:
    """CLS sits at index 0, so under causal attention it sees only itself.

    This is not a defect to fix -- a prefix token that read the sequence would
    feed the future back into every later position at the next layer -- but it is
    why ``--probe-features last`` exists, so it is asserted rather than assumed.
    """
    torch.manual_seed(0)
    encoder = Encoder(16, depth=2, heads=4, mlp="gelu", causal=True).eval()
    valid = torch.ones(2, 9, dtype=torch.bool)
    with torch.no_grad():
        a = encoder(torch.randn(2, 9, 16), valid).cls
        b = encoder(torch.randn(2, 9, 16), valid).cls
    assert torch.allclose(a, b, atol=1e-6)


def test_ar_model_predictions_do_not_see_their_own_target_event() -> None:
    torch.manual_seed(0)
    model = EHRAR(_config()).eval()
    batch = _batch()
    with torch.no_grad():
        base = model(batch)
        bumped = dict(batch)
        codes = batch["code_id"].clone()
        codes[0, 10:] = (codes[0, 10:] + 7) % VOCAB
        bumped["code_id"] = codes
        after = model(bumped)
    # Row 0 contributes targets for positions 0..L-2; only those at index >= 9
    # (whose input window reaches event 10) may move.
    per_row = batch["attention_mask"][0].sum().item() - 1
    assert torch.allclose(base.logits[:9], after.logits[:9], atol=1e-5)
    assert per_row > 9


def test_ar_model_rejects_a_bidirectional_config() -> None:
    with pytest.raises(ValueError, match="causal"):
        EHRAR(_config(causal=False))


# --------------------------------------------------------------------------- #
# The head and the loss
# --------------------------------------------------------------------------- #


def test_tied_head_shares_the_code_table_and_does_not_duplicate_it() -> None:
    model = EHRAR(_config(tie_embeddings=True))
    assert model.head.weight is model.embed.code_emb.weight
    assert model.head.weight.shape == (VOCAB, 32)
    names = [name for name, _ in model.named_parameters()]
    assert sum("code_emb" in name for name in names) == 1
    assert not any(name.startswith("head.proj") for name in names)
    counts = model.n_parameters()
    # Tied: the head owns only its LayerNorm and the output bias.
    assert counts["head"] == 2 * 32 + VOCAB
    assert counts["predictor"] == 0
    out = model(_batch())
    assert out.logits.shape[1] == VOCAB


def test_untied_head_allocates_its_own_projection() -> None:
    model = EHRAR(_config(tie_embeddings=False))
    assert model.head.weight is not model.embed.code_emb.weight
    assert model.head.weight.shape == (VOCAB, 32)
    counts = model.n_parameters()
    assert counts["head"] == 2 * 32 + VOCAB + VOCAB * 32
    assert model(_batch()).logits.shape[1] == VOCAB


def test_next_code_targets_shift_by_one_and_stop_at_padding() -> None:
    codes = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    targets = next_code_targets(codes, valid)
    # Last valid position of each row has no successor -> ignored.
    assert targets.tolist() == [[6, 7, 8, PAD_ID], [10, PAD_ID, PAD_ID, PAD_ID]]


def test_ar_loss_ignores_pad_targets_entirely() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, VOCAB)
    targets = torch.tensor([[6, 7, PAD_ID, PAD_ID], [10, PAD_ID, PAD_ID, PAD_ID]])
    stats = ar_loss(logits, targets)
    assert int(stats["n_targets"]) == 3

    # Same three scored positions, arbitrary garbage everywhere else: identical.
    noisy = logits.clone()
    noisy[0, 2:] = 1e3
    noisy[1, 1:] = -1e3
    assert torch.allclose(stats["ce"], ar_loss(noisy, targets)["ce"], atol=1e-6)

    # And the reference value is a plain cross-entropy over those three rows.
    flat = logits.reshape(-1, VOCAB)[torch.tensor([0, 1, 4])]
    gold = torch.tensor([6, 7, 10])
    expected = torch.nn.functional.cross_entropy(flat, gold)
    assert torch.allclose(stats["ce"], expected, atol=1e-6)


def test_ar_loss_on_an_all_pad_batch_is_zero_and_finite() -> None:
    stats = ar_loss(torch.randn(2, 3, VOCAB), torch.full((2, 3), PAD_ID))
    assert int(stats["n_targets"]) == 0
    assert float(stats["loss"]) == 0.0
    assert torch.isfinite(stats["loss"])


def test_ar_top_k_accuracy_is_exact_on_a_constructed_ranking() -> None:
    logits = torch.zeros(1, 2, 20)
    logits[0, 0, 3] = 5.0  # rank 1 for target 3
    logits[0, 0, 4] = 4.0
    logits[0, 1, 19] = 5.0  # target 4 buried, but inside the top 10 by tie order
    targets = torch.tensor([[3, PAD_ID]])
    stats = ar_loss(logits, targets, top_k=(1, 10))
    assert float(stats["top1"]) == 1.0
    assert float(stats["top10"]) == 1.0


# --------------------------------------------------------------------------- #
# Probing an AR checkpoint
# --------------------------------------------------------------------------- #


def _anchor_frame(cache_dir: Path) -> pl.DataFrame:
    from ehrjepa.eval.history import HistoryReader

    reader = HistoryReader(cache_dir, max_len=None)
    dataset = reader.dataset("train")
    subjects = dataset.index["subject_id"].to_list()[:6]
    return pl.DataFrame(
        {
            "subject_id": list(subjects),
            "anchor_time": [dt.datetime(2200, 1, 1)] * len(subjects),
            "split": ["train"] * len(subjects),
        }
    ).with_columns(pl.col("anchor_time").cast(pl.Datetime("us")))


@pytest.mark.parametrize(
    ("features", "multiple"), [("cls_mean", 2), ("last", 1), ("cls_mean_last", 3)]
)
@pytest.mark.parametrize("layer", ["final", "penultimate"])
@requires_cache
def test_probe_feature_and_layer_options_have_the_documented_width(
    tmp_path: Path, features: str, multiple: int, layer: str
) -> None:
    from ehrjepa.eval.history import HistoryReader

    anchors = _anchor_frame(DEMO_CACHE)
    matrix = probe.embed(
        None,
        DEMO_CACHE,
        anchors,
        vocab_size=HistoryReader(DEMO_CACHE, max_len=None).vocab_size,
        device="cpu",
        features=features,
        layer=layer,
    )
    assert matrix.shape == (anchors.height, multiple * 256)
    assert np.isfinite(matrix).all()


def test_embedding_path_keeps_the_historic_name_for_the_default_pooling() -> None:
    default = probe.embedding_path("cache", "src", "ckpt:a")
    assert default.name == "emb__ckpt-a.parquet"
    other = probe.embedding_path("cache", "src", "ckpt:a", "last", "penultimate")
    assert other.name == "emb__ckpt-a__last__penultimate.parquet"
    assert other != default


def test_penultimate_is_the_stream_entering_the_last_block() -> None:
    torch.manual_seed(0)
    encoder = Encoder(16, depth=3, heads=4, mlp="gelu").eval()
    tokens = torch.randn(2, 7, 16)
    valid = torch.ones(2, 7, dtype=torch.bool)
    with torch.no_grad():
        out = encoder(tokens, valid, return_penultimate=True)
        # Feeding the penultimate stream through the last block and the output
        # norm has to reproduce the final output exactly.
        cos, sin = encoder.rope(8, tokens.device, tokens.dtype)
        stream = torch.cat([out.cls_penultimate[:, None], out.tokens_penultimate], dim=1)
        redone = encoder.norm(
            encoder.blocks[-1](
                stream, cos[None, None], sin[None, None], valid.new_ones(2, 8)[:, None, None, :]
            )
        )
    assert torch.allclose(redone[:, 1:], out.tokens, atol=1e-5)
    assert out.tokens_penultimate.shape == out.tokens.shape


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@requires_cache
def test_five_step_ar_run_on_the_debug_config_is_fast_and_logs(tmp_path: Path) -> None:
    started = time.perf_counter()
    config = load_config(
        DEBUG_CONFIG,
        [
            "objective.kind=ar",
            "run.steps=5",
            "run.log_every=1",
            "run.tensorboard=false",
            f"run.out_dir={tmp_path}",
        ],
    )
    trainer = Trainer(config)
    assert trainer.model_config.causal is True
    final = trainer.train()
    elapsed = time.perf_counter() - started
    assert elapsed < 20.0, f"debug AR run took {elapsed:.1f}s, budget is 20s"
    assert trainer.step == 5
    assert final["ce"] > 0
    assert 0.0 <= final["top1"] <= final["top10"] <= 1.0
    rows = (tmp_path / "metrics.csv").read_text().strip().splitlines()
    assert rows[0].split(",")[:9] == [
        "step",
        "loss",
        "pred_loss",
        "sigreg_tokens",
        "sigreg_cls",
        "ce",
        "top1",
        "top10",
        "lr",
    ]
    assert len(rows) == 6

    # The checkpoint identifies itself as AR, and the probe path picks that up.
    model, max_len = probe.load_encoder(tmp_path / "final.pt")
    assert isinstance(model, EHRAR)
    assert max_len == 64
