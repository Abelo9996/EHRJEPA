"""Config loading, the LR schedule, an end-to-end debug run, and resume equality.

The end-to-end test runs against the committed ``configs/pretrain_debug.yaml`` and
the ``mimic-demo`` cache when one is present; without a cache it is skipped rather
than faked, since a training-loop test that never touches data proves nothing.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
import yaml

from ehrjepa.train.config import apply_overrides, load_config
from ehrjepa.train.pretrain import Trainer, cosine_lr, param_groups

REPO = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = REPO / "configs" / "pretrain_debug.yaml"
DEMO_CACHE = REPO / "data" / "cache" / "mimic-demo"

requires_cache = pytest.mark.skipif(
    not (DEMO_CACHE / "meta.json").exists(), reason="mimic-demo cache is not built"
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_shipped_configs_load_and_have_the_documented_shape() -> None:
    small = load_config(REPO / "configs" / "pretrain_small.yaml")
    assert (small.model["depth"], small.model["dim"], small.model["heads"]) == (6, 256, 4)
    assert (small.model["pred_depth"], small.model["pred_dim"]) == (4, 128)
    assert small.data.max_len == 512
    assert small.run.batch_size == 32
    assert small.optim.lr == pytest.approx(3e-4)
    assert small.run.steps == 2000
    assert small.objective.lambda_sigreg == pytest.approx(0.05)

    debug = load_config(DEBUG_CONFIG)
    assert debug.run.steps == 50


def test_objective_shape_sections_default_to_the_shipped_behaviour() -> None:
    """``predictor``/``target``/``train`` exist, default to off, and reach the model."""
    debug = load_config(DEBUG_CONFIG)
    assert debug.predictor.mask_token_time is True
    assert debug.target.time_features is True
    assert debug.target.span_only is False
    assert debug.train.time_feature_dropout == 0.0
    assert debug.objective.lambda_recon == 0.0

    model = debug.model_config(vocab_size=101)
    assert model.mask_token_time is True
    assert (model.target_time_features, model.target_span_only) == (True, False)
    assert (model.recon_head, model.recon_value_head) == (False, False)


def test_objective_shape_sections_fold_into_the_model_config() -> None:
    """One place to set each flag, and a checkpoint records what it was set to."""
    config = load_config(
        DEBUG_CONFIG,
        [
            "predictor.mask_token_time=false",
            "target.time_features=false",
            "target.span_only=true",
            "train.time_feature_dropout=0.3",
            "objective.lambda_recon=0.1",
            "objective.recon_value=true",
        ],
    )
    model = config.model_config(vocab_size=101)
    assert model.mask_token_time is False
    assert model.target_time_features is False
    assert model.target_span_only is True
    assert model.time_feature_dropout == pytest.approx(0.3)
    assert model.recon_head is True and model.recon_value_head is True
    # The sections round-trip into the payload a checkpoint stores.
    assert config.to_dict()["target"] == {"time_features": False, "span_only": True}

    # No reconstruction weight, no head, even if the value flag is set.
    without = load_config(DEBUG_CONFIG, ["objective.recon_value=true"]).model_config(101)
    assert without.recon_head is False and without.recon_value_head is False


def test_overrides_are_parsed_as_yaml_scalars() -> None:
    raw = {"run": {"steps": 2000}}
    apply_overrides(raw, ["run.steps=5", "optim.lr=1e-5", "run.tensorboard=false", "model.dim=64"])
    assert raw["run"]["steps"] == 5
    assert raw["optim"]["lr"] == pytest.approx(1e-5)
    assert raw["run"]["tensorboard"] is False
    assert raw["model"]["dim"] == 64


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"run": {"stpes": 10}}))
    with pytest.raises(ValueError, match="unknown keys in config section 'run'"):
        load_config(path)


def test_cosine_schedule_warms_up_then_decays() -> None:
    lrs = [cosine_lr(step, 100, 1.0, 10, 0.01) for step in range(100)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert max(lrs) == pytest.approx(1.0)
    assert lrs[10:] == sorted(lrs[10:], reverse=True)
    assert cosine_lr(100, 100, 1.0, 10, 0.01) == pytest.approx(0.01)


def test_param_groups_exclude_norms_biases_and_embeddings_from_decay() -> None:
    from ehrjepa.models import EHRJEPA, EHRJEPAConfig

    model = EHRJEPA(
        EHRJEPAConfig(vocab_size=32, dim=16, depth=1, heads=2, pred_dim=8, pred_depth=1)
    )
    decay, no_decay = param_groups(model, 0.05)
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    decayed = {id(p) for p in decay["params"]}
    assert id(model.embed.code_emb.weight) not in decayed
    assert id(model.encoder.cls_token) not in decayed
    assert id(model.predictor.mask_token) not in decayed
    assert all(p.ndim >= 2 for p in decay["params"])


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def _trainer(tmp_path: Path, steps: int = 5, **overrides: str) -> Trainer:
    args = [
        f"run.out_dir={tmp_path}",
        f"run.steps={steps}",
        "run.device=cpu",
        "run.tensorboard=false",
        *[f"{k}={v}" for k, v in overrides.items()],
    ]
    return Trainer(load_config(DEBUG_CONFIG, args))


@requires_cache
def test_five_step_run_on_the_debug_config_is_fast_and_logs(tmp_path: Path) -> None:
    started = time.perf_counter()
    trainer = _trainer(tmp_path, steps=5)
    final = trainer.train()
    elapsed = time.perf_counter() - started
    assert elapsed < 20.0, f"debug run took {elapsed:.1f}s, budget is 20s"
    assert trainer.step == 5
    assert final["step"] == 5
    assert final["pred_loss"] > 0
    rows = (tmp_path / "metrics.csv").read_text().strip().splitlines()
    assert rows[0].startswith(
        "step,loss,pred_loss,sigreg_tokens,sigreg_cls,recon_loss,recon_value_loss,ce,top1,top10,lr"
    )
    assert len(rows) >= 2
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "config.json").exists()


@requires_cache
def test_the_default_objective_reproduces_its_recorded_loss(tmp_path: Path) -> None:
    """A checksum on the shipped defaults, to the last bit of a float32.

    Every knob added for the second pilot grid defaults to the behaviour this
    number was produced by -- no extra RNG draw, no reordered addition, no head
    that exists but contributes zero. If this moves, an existing run no longer
    reproduces, and that has to be a deliberate decision rather than a surprise.
    """
    final = _trainer(tmp_path, steps=5).train()
    assert final["loss"] == pytest.approx(0.46185019612312317, abs=0.0)
    assert final["pred_loss"] == pytest.approx(0.42000460624694824, abs=0.0)
    assert final["sigreg_tokens"] == pytest.approx(0.3700316250324249, abs=0.0)
    assert final["sigreg_cls"] == pytest.approx(0.4668799340724945, abs=0.0)
    assert final["recon_loss"] == 0.0 and final["recon_value_loss"] == 0.0


@requires_cache
def test_checkpoint_round_trips_model_optimizer_step_config_and_vocab(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, steps=3)
    trainer.train()
    state = torch.load(tmp_path / "final.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 3
    assert state["config"]["run"]["steps"] == 3
    assert state["vocab"]["vocab_size"] == trainer.meta["vocab_size"]
    assert state["vocab"]["tokenizer_version"] == trainer.meta["tokenizer_version"]
    assert set(state) == {"step", "model", "optimizer", "config", "model_config", "vocab", "rng"}


@requires_cache
def test_resume_reproduces_an_uninterrupted_run(tmp_path: Path) -> None:
    """Four steps straight through must equal two steps, save, resume, two more."""
    straight = _trainer(tmp_path / "straight", steps=4)
    straight.train()
    reference = {k: v.clone() for k, v in straight.model.state_dict().items()}

    first = _trainer(tmp_path / "split", steps=2)
    first.train()

    second = _trainer(tmp_path / "split2", steps=4)
    second.load_checkpoint(tmp_path / "split" / "final.pt")
    assert second.step == 2
    second.train()

    for key, expected in reference.items():
        actual = second.model.state_dict()[key]
        assert torch.allclose(actual, expected, atol=1e-5), f"{key} diverged after resume"
