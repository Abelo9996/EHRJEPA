"""``model.target_mode: frozen``, ``model.target_init`` and ``model.init_from``.

Three properties, in the order a run depends on them: the frozen teacher's
weights come from the named checkpoint and stay there, the online stack can be
initialized from the same (or another) checkpoint independently of that, and a
checkpoint of the wrong shape is refused by name rather than by a wall of
``load_state_dict`` shape errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ehrjepa.models import EHRAR, EHRJEPA, EHRJEPAConfig
from ehrjepa.models.latent import EHRNextLatent
from ehrjepa.models.pretrained import check_architecture, load_encoder_weights
from ehrjepa.train.config import load_config
from ehrjepa.train.pretrain import Trainer, param_groups

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
        depth=2,
        heads=4,
        pred_dim=16,
        pred_depth=2,
        pred_heads=2,
        n_freq=8,
    )
    values.update(overrides)
    return EHRJEPAConfig(**values)


def _batch(batch: int = 3, length: int = 24, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "code_id": torch.randint(0, VOCAB, (batch, length), generator=g),
        "value_bin": torch.randint(0, 11, (batch, length), generator=g),
        "value_z": torch.randn(batch, length, generator=g),
        "age": torch.rand(batch, length, generator=g) * 90,
        "log_delta": torch.rand(batch, length, generator=g) * 8,
        "attention_mask": torch.ones(batch, length, dtype=torch.long),
    }


def _write_checkpoint(path: Path, config: EHRJEPAConfig, seed: int = 7) -> Path:
    """A minimal payload with the two keys :mod:`ehrjepa.models.pretrained` reads."""
    torch.manual_seed(seed)
    source = EHRJEPA(config)
    with torch.no_grad():  # move it off its own initialization, so "copied" is visible
        for p in source.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": source.state_dict(), "model_config": vars(config)}, path)
    return path


# --------------------------------------------------------------------------- #
# The frozen teacher
# --------------------------------------------------------------------------- #


def test_frozen_target_loads_the_checkpoint_and_never_moves(tmp_path: Path) -> None:
    source_config = _config()
    ckpt = _write_checkpoint(tmp_path / "source.pt", source_config)
    source = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]

    model = EHRJEPA(_config(target_mode="frozen", target_init=str(ckpt)))
    assert model.uses_target_copy and not model.uses_ema
    for name, p in model.target_encoder.named_parameters():
        assert not p.requires_grad, f"target_encoder.{name} is trainable"
        assert torch.equal(p, source[f"encoder.{name}"])
    for name, p in model.target_embed.named_parameters():
        assert not p.requires_grad, f"target_embed.{name} is trainable"
        assert torch.equal(p, source[f"embed.{name}"])
    # The online stack is *not* the teacher: only ``init_from`` does that.
    assert not torch.equal(
        model.encoder.state_dict()["blocks.0.attn.qkv.weight"],
        source["encoder.blocks.0.attn.qkv.weight"],
    )


def test_a_step_moves_the_student_and_leaves_the_frozen_teacher_alone(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "source.pt", _config())
    model = EHRJEPA(_config(target_mode="frozen", target_init=str(ckpt)))
    before = {k: v.clone() for k, v in model.target_encoder.state_dict().items()}
    before_embed = {k: v.clone() for k, v in model.target_embed.state_dict().items()}
    online_before = model.encoder.state_dict()["blocks.0.attn.qkv.weight"].clone()

    optimizer = torch.optim.AdamW(param_groups(model, 0.05), lr=1e-2)
    batch = _batch()
    mask = batch["attention_mask"].bool()
    target_mask = torch.zeros_like(mask)
    target_mask[:, -6:] = True
    out = model(batch, mask & ~target_mask, target_mask)
    (out.predictions - out.targets).pow(2).mean().backward()
    optimizer.step()

    # ``update_ema`` is called by the trainer for every model; a frozen teacher
    # must ignore it as completely as a shared one does.
    model.update_ema(0.5)

    for key, old in before.items():
        assert torch.equal(model.target_encoder.state_dict()[key], old), key
    for key, old in before_embed.items():
        assert torch.equal(model.target_embed.state_dict()[key], old), key
    assert not torch.equal(model.encoder.state_dict()["blocks.0.attn.qkv.weight"], online_before)
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_frozen_targets_work_for_the_causal_next_latent_model(tmp_path: Path) -> None:
    """The hybrid path: no transformer predictor, targets from ``window_targets``."""
    shape = dict(causal=True, build_predictor=False, horizons=[1, 4])
    ckpt = _write_checkpoint(tmp_path / "ar.pt", _config(causal=True))
    model = EHRNextLatent(_config(target_mode="frozen", target_init=str(ckpt), **shape))
    source = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    assert torch.equal(
        model.target_encoder.state_dict()["blocks.0.attn.qkv.weight"],
        source["encoder.blocks.0.attn.qkv.weight"],
    )
    out = model(_batch())
    assert out.predictions.shape == out.targets.shape and out.predictions.numel() > 0


def test_the_optimizer_never_receives_a_frozen_target_parameter(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "source.pt", _config())
    model = EHRJEPA(_config(target_mode="frozen", target_init=str(ckpt)))
    owned = {id(p) for group in param_groups(model, 0.05) for p in group["params"]}
    for p in list(model.target_encoder.parameters()) + list(model.target_embed.parameters()):
        assert id(p) not in owned


# --------------------------------------------------------------------------- #
# Initializing the student
# --------------------------------------------------------------------------- #


def test_init_from_copies_the_online_stack_and_leaves_it_trainable(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "source.pt", _config(causal=True))
    source = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    model = EHRJEPA(_config(causal=True, init_from=str(ckpt)))
    for name, p in model.encoder.named_parameters():
        assert torch.equal(p, source[f"encoder.{name}"])
        assert p.requires_grad
    for name, p in model.embed.named_parameters():
        assert torch.equal(p, source[f"embed.{name}"])


def test_init_from_and_target_init_together_start_student_and_teacher_equal(
    tmp_path: Path,
) -> None:
    ckpt = _write_checkpoint(tmp_path / "source.pt", _config(causal=True))
    model = EHRJEPA(
        _config(causal=True, target_mode="frozen", target_init=str(ckpt), init_from=str(ckpt))
    )
    online, target = model.encoder.state_dict(), model.target_encoder.state_dict()
    for key, value in online.items():
        assert torch.equal(value, target[key]), key


def test_init_from_works_for_the_ar_model(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "source.pt", _config(causal=True))
    source = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    model = EHRAR(_config(causal=True, init_from=str(ckpt)))
    assert torch.equal(
        model.encoder.state_dict()["blocks.0.attn.qkv.weight"],
        source["encoder.blocks.0.attn.qkv.weight"],
    )


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_mismatched_width_is_refused_by_name(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "wide.pt", _config(dim=64, heads=4))
    with pytest.raises(ValueError, match=r"target_init.*dim=64.*this model: 32"):
        EHRJEPA(_config(dim=32, target_mode="frozen", target_init=str(ckpt)))
    with pytest.raises(ValueError, match=r"init_from.*dim=64"):
        EHRJEPA(_config(dim=32, init_from=str(ckpt)))


@pytest.mark.parametrize(
    "overrides", [{"depth": 4}, {"vocab_size": 128}, {"n_freq": 16}, {"mlp": "gelu"}]
)
def test_every_shape_field_is_checked(tmp_path: Path, overrides: dict) -> None:
    ckpt = _write_checkpoint(tmp_path / "other.pt", _config(**overrides))
    with pytest.raises(ValueError, match="different architecture"):
        EHRJEPA(_config(target_mode="frozen", target_init=str(ckpt)))


def test_a_causal_difference_is_a_note_not_an_error(tmp_path: Path, capsys) -> None:
    ckpt = _write_checkpoint(tmp_path / "causal.pt", _config(causal=True))
    EHRJEPA(_config(causal=False, target_mode="frozen", target_init=str(ckpt)))
    assert "causal=True" in capsys.readouterr().out


def test_a_missing_checkpoint_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nowhere.pt"):
        EHRJEPA(_config(target_mode="frozen", target_init=str(tmp_path / "nowhere.pt")))


def test_target_init_without_frozen_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only read by target_mode='frozen'"):
        _config(target_mode="ema", target_init="whatever.pt")


def test_frozen_mode_without_a_target_init_is_rejected_at_config_load() -> None:
    """The model itself allows it (that is what ``for_reload`` builds); a *run* does not."""
    config = load_config(DEBUG_CONFIG, ["model.target_mode=frozen"])
    with pytest.raises(ValueError, match="model.target_init"):
        config.model_config(vocab_size=101)


def test_for_reload_drops_the_initialization_sources_but_keeps_the_shape() -> None:
    config = _config(target_mode="frozen", target_init="a.pt", init_from="b.pt")
    reload = config.for_reload()
    assert reload.target_init is None and reload.init_from is None
    assert reload.code_init == "random"
    assert reload.target_mode == "frozen" and reload.dim == config.dim
    # Buildable without either file present -- which is the whole point.
    assert EHRJEPA(reload).target_encoder is not None


def test_check_architecture_needs_the_fields_it_compares() -> None:
    with pytest.raises(ValueError, match="records no"):
        check_architecture(_config(), {"dim": 32}, "x.pt", "target_init")


def test_a_checkpoint_without_encoder_weights_is_refused(tmp_path: Path) -> None:
    config = _config()
    path = tmp_path / "headless.pt"
    torch.save({"model": {"head.bias": torch.zeros(3)}, "model_config": vars(config)}, path)
    model = EHRJEPA(config)
    with pytest.raises(ValueError, match="no embed.. weights"):
        load_encoder_weights(model.embed, model.encoder, path, config, "target_init")


def test_a_payload_that_is_not_a_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "junk.pt"
    torch.save({"hello": 1}, path)
    with pytest.raises(ValueError, match="not an ehrjepa training checkpoint"):
        EHRJEPA(_config(target_mode="frozen", target_init=str(path)))


# --------------------------------------------------------------------------- #
# Against a real checkpoint
# --------------------------------------------------------------------------- #


@requires_cache
def test_a_debug_config_checkpoint_round_trips_into_a_frozen_teacher(tmp_path: Path) -> None:
    """Train two steps on the shipped debug config, then use its ``final.pt`` as a teacher."""
    source_dir = tmp_path / "source"
    trainer = Trainer(
        load_config(
            DEBUG_CONFIG,
            [f"run.out_dir={source_dir}", "run.steps=2", "run.device=cpu", "run.tensorboard=false"],
        )
    )
    trainer.train()
    final = source_dir / "final.pt"
    assert final.exists()

    student = Trainer(
        load_config(
            DEBUG_CONFIG,
            [
                f"run.out_dir={tmp_path / 'student'}",
                "run.steps=1",
                "run.device=cpu",
                "run.tensorboard=false",
                "model.target_mode=frozen",
                f"model.target_init={final}",
                f"model.init_from={final}",
            ],
        )
    )
    teacher = trainer.model.encoder.state_dict()
    for key, value in student.model.target_encoder.state_dict().items():
        assert torch.equal(value, teacher[key]), key
    for key, value in student.model.encoder.state_dict().items():
        assert torch.equal(value, teacher[key]), key

    student.train()
    for key, value in student.model.target_encoder.state_dict().items():
        assert torch.equal(value, teacher[key]), f"{key} moved during training"
    assert not torch.equal(
        student.model.encoder.state_dict()["blocks.0.attn.qkv.weight"],
        teacher["blocks.0.attn.qkv.weight"],
    )
