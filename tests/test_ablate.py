"""The ablation runner's planning, resume and rendering -- everything but the GPU.

``run_grid`` itself shells out to the trainer and the evaluation harness and
takes hours, so what is tested here is the part that decides *what* to run: the
budget arithmetic, the skip-what-is-done rule, and the table. Those are the parts
that, if wrong, waste a night rather than raising.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("ablate", REPO / "scripts" / "ablate.py")
assert _spec and _spec.loader
ablate = importlib.util.module_from_spec(_spec)
sys.modules["ablate"] = ablate
_spec.loader.exec_module(ablate)


def _grid_file(tmp_path: Path, **extra) -> Path:
    payload = {
        "name": "unit-grid",
        "base": "configs/pretrain_pilot.yaml",
        "source": "desynpuf-s1",
        "budget_tokens": 3_276_800,
        "docs_root": str(tmp_path / "docs"),
        "runs_root": str(tmp_path / "runs"),
        "runs": [
            {"name": "ar", "overrides": {"objective.kind": "ar", "model.causal": True}},
            {"name": "jepa_ema", "overrides": {"model.target_mode": "ema"}},
        ],
    }
    payload.update(extra)
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


# --------------------------------------------------------------------------- #
# Budget arithmetic
# --------------------------------------------------------------------------- #


def test_steps_are_the_ceiling_of_the_budget_over_the_nominal_window() -> None:
    assert ablate.steps_for(12_000_000, 64, 256) == 733  # 16,384 slots per step
    assert ablate.steps_for(12_000_000, 32, 512) == 733  # same product, same steps
    assert ablate.steps_for(12_000_000, 32, 256) == 1465  # half the window, twice the steps
    assert ablate.steps_for(1, 64, 256) == 1  # never zero


def test_steps_reject_a_degenerate_shape() -> None:
    with pytest.raises(ValueError):
        ablate.steps_for(1000, 0, 256)
    # accum_steps <= 0 is clamped to 1, mirroring the trainer's own
    # ``max(1, cfg.optim.accum_steps)`` -- it is not a degenerate shape.
    assert ablate.steps_for(1000, 32, 256, accum_steps=0) == ablate.steps_for(1000, 32, 256)


def test_steps_account_for_gradient_accumulation() -> None:
    """A halved batch with 2-step accumulation must not double the real budget.

    The trainer's ``step`` counter advances once per *optimizer* step, which
    runs ``accum_steps`` micro-batches first (``Trainer.train``) -- so a cell
    trading ``batch_size: 64`` for ``batch_size: 32`` + ``accum_steps: 2`` needs
    half as many optimizer steps for the same nominal tokens, not the same
    number computed as if accumulation did not exist.
    """
    assert ablate.steps_for(32_768_000, 32, 512, accum_steps=2) == ablate.steps_for(
        32_768_000, 64, 512
    )
    assert ablate.steps_for(32_768_000, 32, 512, accum_steps=1) == 2 * ablate.steps_for(
        32_768_000, 32, 512, accum_steps=2
    )


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def test_dry_run_plan_resolves_every_cell_from_the_base_config(tmp_path: Path) -> None:
    grid = ablate.load_grid(_grid_file(tmp_path))
    entries = ablate.plan(grid)
    assert [e["run"] for e in entries] == ["ar", "jepa_ema"]
    for entry in entries:
        assert entry["batch_size"] == 64
        assert entry["max_len"] == 256
        assert entry["steps"] == 200
        assert entry["tokens"] == 200 * 64 * 256
        assert entry["done"] is False
        assert entry["out_dir"].endswith("/unit-grid/" + entry["run"])
    assert entries[0]["objective"] == "ar"
    # The JEPA knobs are blanked on the AR row rather than inherited from base.
    assert entries[0]["target_mode"] is None
    assert entries[0]["lambda_sigreg"] is None
    assert entries[0]["p_future"] is None
    assert entries[1]["objective"] == "jepa"
    assert entries[1]["target_mode"] == "ema"


def _finished_run(root: Path, name: str, *, kind: str = "jepa", steps: int = 2930) -> Path:
    """A directory shaped like a finished cell: ``config.json``, ``metrics.csv``, ``final.pt``."""
    out = root / name
    out.mkdir(parents=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "data": {"max_len": 256},
                "model": {"target_mode": "ema"},
                "objective": {"kind": kind, "lambda_sigreg": 0.05},
                "masking": {"p_future": 0.6},
                "run": {"steps": steps, "batch_size": 64},
            }
        )
    )
    (out / "metrics.csv").write_text("step,loss,tokens_per_s\n1,9.0,100\n2,6.5,200\n")
    (out / "final.pt").write_bytes(b"")
    return out / "final.pt"


def test_a_reused_checkpoint_is_planned_from_the_run_that_trained_it(tmp_path: Path) -> None:
    """A ``reuse_checkpoint`` row describes the training that actually happened.

    Steps, budget and the JEPA knobs come out of the other run's ``config.json``,
    not out of this grid's base -- and the cell costs no tokens.
    """
    checkpoint = _finished_run(tmp_path / "elsewhere", "ar", kind="ar", steps=2930)
    path = _grid_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["runs"].append({"name": "ar_last", "reuse_checkpoint": str(checkpoint)})
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    entries = ablate.plan(ablate.load_grid(path))
    reused = entries[-1]
    assert reused["reuse"] is True
    assert reused["checkpoint"] == str(checkpoint)
    assert reused["out_dir"] == str(checkpoint.parent)
    assert reused["steps"] == 2930
    assert reused["tokens"] == 2930 * 64 * 256
    assert reused["objective"] == "ar"
    # An AR cell's JEPA columns stay blank even when read back from disk.
    assert reused["target_mode"] is None and reused["p_future"] is None
    assert entries[0]["reuse"] is False

    # It is planned as REUSE and contributes nothing to the outstanding budget.
    ablate.main([str(path), "--dry-run", "--only", "ar_last"])


def test_a_reused_checkpoint_missing_locally_still_dry_runs(tmp_path: Path, capsys) -> None:
    """A ``reuse_checkpoint`` that lives on another machine must not crash planning.

    A grid that re-scores a GPU box's checkpoints has to be plannable (and
    ``--dry-run``-able) on a machine that has never seen those files -- only
    actually evaluating the cell should require the checkpoint to exist.
    """
    missing = tmp_path / "elsewhere" / "ar" / "final.pt"
    path = _grid_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["runs"] = [{"name": "ar_base_s0", "reuse_checkpoint": str(missing)}]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    entries = ablate.plan(ablate.load_grid(path))
    entry = entries[0]
    assert entry["reuse"] is True
    assert entry["missing"] is True
    assert entry["checkpoint"] == str(missing)
    assert entry["steps"] == 0 and entry["tokens"] == 0

    assert ablate.main([str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "REUSE?" in out
    assert "checkpoint not found locally" in out


def test_a_reused_cell_costs_no_training_budget(tmp_path: Path, capsys) -> None:
    checkpoint = _finished_run(tmp_path / "elsewhere", "jepa_ema")
    path = _grid_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["runs"] = [{"name": "jepa_ema_mean", "reuse_checkpoint": str(checkpoint)}]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    ablate.main([str(path), "--dry-run"])
    out = capsys.readouterr().out
    assert "REUSE" in out
    assert "total outstanding: 0 tokens" in out


def test_a_reused_cell_may_not_also_carry_overrides(tmp_path: Path) -> None:
    """Overrides are training knobs; a row that trains nothing must not pretend."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base": "b",
                "runs": [
                    {"name": "a", "reuse_checkpoint": "runs/x/final.pt", "overrides": {"a.b": 1}}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="overrides do nothing"):
        ablate.load_grid(path)


def test_a_per_run_budget_overrides_the_grid_default(tmp_path: Path) -> None:
    path = _grid_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["runs"][0]["budget_tokens"] = 1_638_400
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    entries = ablate.plan(ablate.load_grid(path))
    assert entries[0]["steps"] == 100
    assert entries[1]["steps"] == 200


def test_plan_spends_the_same_tokens_under_gradient_accumulation(tmp_path: Path) -> None:
    """A run that halves its batch and doubles ``accum_steps`` costs the same budget.

    Each optimizer step still consumes ``batch_size x max_len x accum_steps``
    nominal tokens -- half the batch times twice the accumulation is the same
    per-step total -- so a cell overriding to ``batch_size: 32`` with
    ``optim.accum_steps: 2`` must plan to the *same* token total and the *same*
    step count as one left at the unaccumulated ``batch_size: 64`` default, not
    to a step count computed as if accumulation did not exist (which would
    silently double its real token spend).
    """
    path = _grid_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["runs"][0]["overrides"]["run.batch_size"] = 32
    raw["runs"][0]["overrides"]["optim.accum_steps"] = 2
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    entries = ablate.plan(ablate.load_grid(path))
    accumulated, plain = entries[0], entries[1]
    assert accumulated["batch_size"] == 32
    assert accumulated["tokens"] == plain["tokens"]
    assert accumulated["steps"] == plain["steps"]


def test_train_one_passes_the_ckpt_every_override(tmp_path: Path) -> None:
    """A trained cell checkpoints periodically instead of ``ckpt_every=0``."""
    grid = ablate.load_grid(_grid_file(tmp_path, ckpt_every=777))
    entry = ablate.plan(grid)[0]
    captured: dict[str, list] = {}

    def fake_spawn(command, log) -> None:
        captured["command"] = list(command)

    real_spawn, real_final = ablate._spawn, ablate._final_metrics
    ablate._spawn = fake_spawn  # type: ignore[assignment]
    ablate._final_metrics = lambda out_dir: {"loss": 1.0}  # type: ignore[assignment]
    try:
        log = ablate.Log(tmp_path / "log.txt")
        ablate.train_one(grid, entry, log)
        log.close()
    finally:
        ablate._spawn, ablate._final_metrics = real_spawn, real_final

    assert "run.ckpt_every=777" in captured["command"]
    assert "run.ckpt_every=0" not in captured["command"]
    assert not any(str(c).startswith("run.resume=") for c in captured["command"])


def test_train_one_resumes_from_a_latest_checkpoint_without_a_final(tmp_path: Path) -> None:
    """A ``latest.pt`` with no ``final.pt`` means the cell was mid-run -- resume it."""
    grid = ablate.load_grid(_grid_file(tmp_path))
    entry = ablate.plan(grid)[0]
    out_dir = Path(entry["out_dir"])
    out_dir.mkdir(parents=True)
    latest = out_dir / "latest.pt"
    latest.write_bytes(b"not a real checkpoint")
    captured: dict[str, list] = {}

    def fake_spawn(command, log) -> None:
        captured["command"] = list(command)

    real_spawn, real_final = ablate._spawn, ablate._final_metrics
    ablate._spawn = fake_spawn  # type: ignore[assignment]
    ablate._final_metrics = lambda out_dir: {"loss": 1.0}  # type: ignore[assignment]
    try:
        log = ablate.Log(tmp_path / "log.txt")
        ablate.train_one(grid, entry, log)
        log.close()
    finally:
        ablate._spawn, ablate._final_metrics = real_spawn, real_final

    assert f"run.resume={latest}" in captured["command"]
    log_text = (tmp_path / "log.txt").read_text()
    assert f"resuming {entry['run']}" in log_text


def test_train_one_does_not_resume_when_final_already_exists(tmp_path: Path) -> None:
    """``final.pt`` present means the cell is done, even with a stray ``latest.pt``."""
    grid = ablate.load_grid(_grid_file(tmp_path))
    entry = ablate.plan(grid)[0]
    out_dir = Path(entry["out_dir"])
    out_dir.mkdir(parents=True)
    (out_dir / "latest.pt").write_bytes(b"not a real checkpoint")
    (out_dir / "final.pt").write_bytes(b"not a real checkpoint either")
    (out_dir / "metrics.csv").write_text("step,loss\n1,1.0\n")

    # The final.pt + metrics.csv reuse path short-circuits before ever spawning.
    def fail_spawn(command, log) -> None:
        raise AssertionError("should not train a cell that already has final.pt")

    real_spawn = ablate._spawn
    ablate._spawn = fail_spawn  # type: ignore[assignment]
    try:
        log = ablate.Log(tmp_path / "log.txt")
        result = ablate.train_one(grid, entry, log)
        log.close()
    finally:
        ablate._spawn = real_spawn

    assert result["wall_s"] == 0.0


def test_plan_skips_runs_already_in_the_summary(tmp_path: Path) -> None:
    grid = ablate.load_grid(_grid_file(tmp_path))
    grid.summary_json.parent.mkdir(parents=True, exist_ok=True)
    grid.summary_json.write_text(
        json.dumps({"grid": "unit-grid", "runs": [{"run": "ar", "auroc": {}}]})
    )
    entries = ablate.plan(grid)
    assert [e["done"] for e in entries] == [True, False]


def test_main_dry_run_prints_the_plan_and_touches_nothing(tmp_path: Path, capsys) -> None:
    path = _grid_file(tmp_path)
    assert ablate.main([str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "RUN" in out and "ar" in out and "jepa_ema" in out
    assert "6,553,600 tokens" in out
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "docs" / "unit-grid" / "summary.json").exists()


def test_main_dry_run_marks_finished_cells_as_skipped(tmp_path: Path, capsys) -> None:
    path = _grid_file(tmp_path)
    grid = ablate.load_grid(path)
    grid.summary_json.parent.mkdir(parents=True, exist_ok=True)
    grid.summary_json.write_text(json.dumps({"runs": [{"run": "ar"}, {"run": "jepa_ema"}]}))
    ablate.main([str(path), "--dry-run"])
    out = capsys.readouterr().out
    assert out.count("SKIP (done)") == 2
    assert "total outstanding: 0 tokens" in out
    # --force ignores the summary and plans everything again.
    ablate.main([str(path), "--dry-run", "--force"])
    assert capsys.readouterr().out.count("RUN") >= 2


def test_only_restricts_the_plan(tmp_path: Path, capsys) -> None:
    ablate.main([str(_grid_file(tmp_path)), "--dry-run", "--only", "jepa_ema"])
    out = capsys.readouterr().out
    assert "jepa_ema" in out
    assert "\n  RUN         ar " not in out


# --------------------------------------------------------------------------- #
# Evaluation: the full-held-out path and fresh baselines
# --------------------------------------------------------------------------- #


def _fake_results(eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "results.json").write_text(json.dumps({"tasks": {}, "models": {}}))


def test_eval_one_omits_the_subject_limit_flag_when_none(tmp_path: Path) -> None:
    """``eval_subject_limit: null`` must score the full held-out split.

    The harness's own default (``--eval-subject-limit`` absent) is "no limit",
    so the runner has to omit the flag rather than invent a value -- passing
    ``--eval-subject-limit 0`` or the literal string ``None`` would either error
    or silently keep every subject for the wrong reason.
    """
    grid = ablate.load_grid(_grid_file(tmp_path, eval_subject_limit=None))
    entry = ablate.plan(grid)[0]
    captured: dict[str, list] = {}

    def fake_spawn(command, log) -> None:
        captured["command"] = list(command)
        _fake_results(Path(entry["eval_dir"]))

    real_spawn = ablate._spawn
    ablate._spawn = fake_spawn  # type: ignore[assignment]
    try:
        log = ablate.Log(tmp_path / "log.txt")
        ablate.eval_one(grid, entry, with_controls=False, log=log)
        log.close()
    finally:
        ablate._spawn = real_spawn

    assert "--eval-subject-limit" not in captured["command"]
    assert "--eval-subject-seed" not in captured["command"]


def test_eval_one_keeps_the_subject_limit_flag_when_set(tmp_path: Path) -> None:
    grid = ablate.load_grid(_grid_file(tmp_path, eval_subject_limit=3000))
    entry = ablate.plan(grid)[0]
    captured: dict[str, list] = {}

    def fake_spawn(command, log) -> None:
        captured["command"] = list(command)
        _fake_results(Path(entry["eval_dir"]))

    real_spawn = ablate._spawn
    ablate._spawn = fake_spawn  # type: ignore[assignment]
    try:
        log = ablate.Log(tmp_path / "log.txt")
        ablate.eval_one(grid, entry, with_controls=False, log=log)
        log.close()
    finally:
        ablate._spawn = real_spawn

    assert "--eval-subject-limit" in captured["command"]
    assert "3000" in captured["command"]


def test_baselines_needed_without_reuse_predictions_until_cached(tmp_path: Path) -> None:
    grid = ablate.load_grid(_grid_file(tmp_path))
    assert grid.reuse_predictions is None
    assert ablate.baselines_needed(grid, {}) is True
    assert ablate.baselines_needed(grid, {"lr": {}, "gbm": {}}) is False
    assert ablate.baselines_needed(grid, {"lr": {}}) is True


def test_baselines_not_needed_when_reuse_predictions_is_set(tmp_path: Path) -> None:
    grid = ablate.load_grid(
        _grid_file(tmp_path, reuse_predictions="docs/experiments/x/predictions.parquet")
    )
    assert ablate.baselines_needed(grid, {}) is False


def test_eval_one_adds_reuse_models_when_baselines_are_needed(tmp_path: Path) -> None:
    """``with_baselines`` rides ``lr``/``gbm`` along on this cell's command."""
    grid = ablate.load_grid(_grid_file(tmp_path))
    entry = ablate.plan(grid)[0]
    captured: dict[str, list] = {}

    def fake_spawn(command, log) -> None:
        captured["command"] = list(command)
        _fake_results(Path(entry["eval_dir"]))

    real_spawn = ablate._spawn
    ablate._spawn = fake_spawn  # type: ignore[assignment]
    try:
        log = ablate.Log(tmp_path / "log.txt")
        ablate.eval_one(grid, entry, with_controls=False, log=log, with_baselines=True)
        log.close()
    finally:
        ablate._spawn = real_spawn

    models_arg = captured["command"][captured["command"].index("--models") + 1]
    assert models_arg.split(",")[:2] == ["lr", "gbm"]


def test_baselines_for_caches_fresh_lr_gbm_without_reuse_predictions(tmp_path: Path) -> None:
    """No ``reuse_predictions`` file: ``lr``/``gbm`` are cached from ``fresh`` instead.

    They land under their bare model name -- not ``lr@ar`` -- because, unlike
    ``random_init``, they do not depend on which cell happened to compute them.
    """
    grid = ablate.load_grid(_grid_file(tmp_path))
    fresh = {
        "lr": {"mortality_365d": 0.60},
        "gbm": {"mortality_365d": 0.62},
        "ckpt:ar": {"mortality_365d": 0.70},
    }
    stored = ablate.baselines_for(grid, fresh, "ar")
    assert stored["lr"]["mortality_365d"] == 0.60
    assert stored["gbm"]["mortality_365d"] == 0.62
    assert "ckpt:ar" not in stored

    # Cached to disk, and picked up on a later call with no `fresh` at all.
    again = ablate.baselines_for(grid)
    assert again["lr"]["mortality_365d"] == 0.60


def test_baselines_for_does_not_fabricate_lr_gbm_when_reuse_predictions_is_set(
    tmp_path: Path,
) -> None:
    """A grid with a real ``reuse_predictions`` file must not also self-fit."""
    grid = ablate.load_grid(
        _grid_file(
            tmp_path, reuse_predictions="docs/experiments/does-not-exist/predictions.parquet"
        )
    )
    fresh = {"lr": {"mortality_365d": 0.60}, "ckpt:ar": {"mortality_365d": 0.70}}
    stored = ablate.baselines_for(grid, fresh, "ar")
    assert "lr" not in stored


def test_render_summary_describes_the_full_held_out_split_when_limit_is_none() -> None:
    payload = {
        "grid": "g",
        "base": "configs/pretrain_scale.yaml",
        "source": "desynpuf-s1",
        "updated": "2026-09-07T00:00:00+00:00",
        "eval": {
            "bootstrap": 200,
            "eval_subject_limit": None,
            "eval_subject_seed": 0,
            "probe_features": "auto",
            "probe_layer": "final",
        },
        "runs": [],
    }
    text = ablate.render_summary(payload)
    assert "the full held-out split" in text
    assert "None-subject" not in text
    assert "None" not in text.split("\n")[2]


# --------------------------------------------------------------------------- #
# Grid file validation
# --------------------------------------------------------------------------- #


def test_unknown_grid_keys_and_duplicate_runs_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"base": "b", "runs": [{"name": "a"}], "nonsense": 1}))
    with pytest.raises(ValueError, match="unknown grid keys"):
        ablate.load_grid(path)

    path.write_text(yaml.safe_dump({"base": "b", "runs": [{"name": "a"}, {"name": "a"}]}))
    with pytest.raises(ValueError, match="duplicate run name"):
        ablate.load_grid(path)

    path.write_text(yaml.safe_dump({"base": "b", "runs": [{"name": "a", "oops": 1}]}))
    with pytest.raises(ValueError, match="unknown keys"):
        ablate.load_grid(path)

    path.write_text(yaml.safe_dump({"base": "b", "runs": []}))
    with pytest.raises(ValueError, match="no runs"):
        ablate.load_grid(path)


def test_booleans_survive_the_trip_through_override_strings(tmp_path: Path) -> None:
    grid = ablate.load_grid(_grid_file(tmp_path))
    assert "model.causal=true" in grid.runs[0].override_strings()


def test_the_shipped_grids_parse_and_plan(tmp_path: Path) -> None:
    for name in ("micro_desynpuf", "pilot_desynpuf"):
        grid = ablate.load_grid(REPO / "configs" / "grids" / f"{name}.yaml")
        entries = ablate.plan(grid)
        assert entries, name
        assert all(e["steps"] > 0 for e in entries)
        assert entries[0]["objective"] == "ar"
    pilot = ablate.load_grid(REPO / "configs" / "grids" / "pilot_desynpuf.yaml")
    assert [item.name for item in pilot.runs] == [
        "ar",
        "jepa_ema",
        "jepa_ema_nosig",
        "jepa_shared_sig",
        "jepa_ema_future",
        "jepa_ema_block",
    ]
    tokens = {e["tokens"] for e in ablate.plan(pilot)}
    assert len(tokens) == 1, "every pilot cell must share one budget"
    assert tokens == {ablate.steps_for(pilot.budget_tokens, 64, 256) * 64 * 256}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_summary_markdown_has_a_row_per_run_and_a_column_per_task() -> None:
    payload = {
        "grid": "g",
        "base": "configs/pretrain_pilot.yaml",
        "source": "desynpuf-s1",
        "updated": "2026-09-03T00:00:00+00:00",
        "eval": {
            "bootstrap": 200,
            "eval_subject_limit": 3000,
            "eval_subject_seed": 0,
            "probe_features": "cls_mean",
            "probe_layer": "final",
        },
        "runs": [
            {
                "run": "ar",
                "objective": "ar",
                "pooling": "last@final",
                "target_mode": None,
                "lambda_sigreg": None,
                "p_future": None,
                "steps": 733,
                "tokens": 12009472,
                "final": {"loss": 6.5, "ce": 6.5, "top1": 0.1, "effective_rank": 40.0},
                "auroc": {"mortality_365d": 0.55, "inpatient_365d": 0.71},
            },
            {
                "run": "jepa_ema",
                "objective": "jepa",
                "pooling": "mean@final",
                "target_mode": "ema",
                "lambda_sigreg": 0.05,
                "p_future": 0.6,
                "steps": 733,
                "tokens": 12009472,
                "final": {"loss": 0.4, "pred_loss": 0.38, "cos_gap": 0.02},
                "auroc": {"mortality_365d": 0.56, "inpatient_365d": 0.70},
            },
        ],
        "baselines": {"gbm": {"mortality_365d": 0.57, "inpatient_365d": 0.74}},
    }
    text = ablate.render_summary(payload)
    assert "`ar`" in text and "`jepa_ema`" in text
    assert "mortality_365d" in text and "inpatient_365d" in text
    assert "0.55" in text and "0.7" in text
    # Blank JEPA knobs render as an em-dash placeholder, not as "None".
    assert "None" not in text
    assert "## Reference models" in text and "`gbm`" in text
    # Every row says which pooling produced its AUROCs, because in one grid they
    # differ: a causal arm is read at `last`, a bidirectional one at `mean`.
    assert "last@final" in text and "mean@final" in text
    header, rule = text.splitlines()[6], text.splitlines()[7]
    assert header.count("|") == rule.count("|")


# --------------------------------------------------------------------------- #
# Evaluation modes
# --------------------------------------------------------------------------- #


def test_a_grid_defaults_to_probe_only_evaluation(tmp_path: Path) -> None:
    """The historical behaviour, unchanged: one probe row per cell, no ft flags."""
    grid = ablate.load_grid(_grid_file(tmp_path))
    assert grid.eval_modes == ("probe",)
    first, second = ablate.plan(grid)
    assert first["eval_models"] == ["random_init", f"ckpt:{first['checkpoint']}"]
    assert second["eval_models"] == [f"ckpt:{second['checkpoint']}"]


def test_eval_modes_add_a_finetune_row_and_its_control(tmp_path: Path) -> None:
    grid = ablate.load_grid(_grid_file(tmp_path, eval_modes=["probe", "finetune"]))
    assert grid.eval_modes == ("probe", "finetune")
    entry = ablate.plan(grid)[0]
    assert entry["eval_models"] == [
        "random_init",
        "ft_random",
        f"ckpt:{entry['checkpoint']}",
        f"ft:{entry['checkpoint']}",
    ]
    # A finetune-only grid asks for neither the frozen probe nor its control.
    only = ablate.load_grid(_grid_file(tmp_path, eval_modes=["finetune"]))
    assert ablate.plan(only)[0]["eval_models"] == [
        "ft_random",
        f"ft:{entry['checkpoint']}",
    ]


def test_unknown_eval_modes_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval_modes"):
        ablate.load_grid(_grid_file(tmp_path, eval_modes=["probe", "linear_head"]))
    with pytest.raises(ValueError, match="eval_modes"):
        ablate.load_grid(_grid_file(tmp_path, eval_modes=[]))


def test_dry_run_lists_the_finetune_rows(tmp_path: Path, capsys) -> None:
    """``--dry-run`` has to show the rows a mode adds, or the plan understates itself."""
    path = _grid_file(tmp_path, eval_modes=["probe", "finetune"])
    assert ablate.main([str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("eval      ft:") == 2, out
    assert "eval      ft_random" in out
    assert "eval      ckpt:" in out

    # A probe-only grid prints exactly what it printed before the modes existed.
    assert ablate.main([str(_grid_file(tmp_path)), "--dry-run"]) == 0
    assert "eval " not in capsys.readouterr().out


def test_eval_one_passes_the_finetune_flags_only_when_a_mode_asks(tmp_path: Path) -> None:
    def command_for(**extra) -> list[str]:
        grid = ablate.load_grid(_grid_file(tmp_path, **extra))
        entry = ablate.plan(grid)[0]
        captured: dict[str, list] = {}

        def fake_spawn(command, log) -> None:
            captured["command"] = list(command)
            _fake_results(Path(entry["eval_dir"]))

        real_spawn = ablate._spawn
        ablate._spawn = fake_spawn  # type: ignore[assignment]
        try:
            log = ablate.Log(tmp_path / "log.txt")
            ablate.eval_one(grid, entry, with_controls=True, log=log)
            log.close()
        finally:
            ablate._spawn = real_spawn
        return captured["command"]

    probe_only = command_for()
    assert "--ft-epochs" not in probe_only
    assert "--ft-balanced" not in probe_only

    tuned = command_for(eval_modes=["finetune"], ft_epochs=3, ft_balanced=True, ft_batch=16)
    assert tuned[tuned.index("--ft-epochs") + 1] == "3"
    assert tuned[tuned.index("--ft-batch") + 1] == "16"
    assert "--ft-balanced" in tuned
    models = tuned[tuned.index("--models") + 1].split(",")
    assert models[0] == "ft_random" and models[1].startswith("ft:")


def test_a_cell_is_done_only_when_every_mode_has_a_row(tmp_path: Path) -> None:
    """Adding a mode to a finished grid re-evaluates its cells for that mode."""
    both = ablate.load_grid(_grid_file(tmp_path, eval_modes=["probe", "finetune"]))
    both.summary_json.parent.mkdir(parents=True, exist_ok=True)
    both.summary_json.write_text(json.dumps({"runs": [{"run": "ar", "mode": "probe"}]}))
    assert [e["done"] for e in ablate.plan(both)] == [False, False]

    both.summary_json.write_text(
        json.dumps({"runs": [{"run": "ar", "mode": "probe"}, {"run": "ar", "mode": "finetune"}]})
    )
    assert [e["done"] for e in ablate.plan(both)] == [True, False]

    # A row written before `eval_modes` existed carries no mode and is a probe row.
    probe_only = ablate.load_grid(_grid_file(tmp_path))
    probe_only.summary_json.write_text(json.dumps({"runs": [{"run": "ar"}]}))
    assert [e["done"] for e in ablate.plan(probe_only)] == [True, False]


def test_a_row_reports_the_model_its_mode_names(tmp_path: Path) -> None:
    entry = {
        "run": "ar",
        "objective": "ar",
        "checkpoint": "runs/g/ar/final.pt",
        "reuse": False,
        "target_mode": None,
        "lambda_sigreg": None,
        "p_future": None,
        "steps": 10,
        "tokens": 100,
        "batch_size": 64,
        "max_len": 256,
        "overrides": {},
    }
    scored = {
        "ckpt:ar": {"mortality_365d": 0.70},
        "ft:ar": {"mortality_365d": 0.75},
    }
    pooling = {"ckpt:ar": "last@final", "ft:ar": "last@final"}
    probed = ablate._row(entry, {}, scored, pooling, "probe")
    tuned = ablate._row(entry, {}, scored, pooling, "finetune")
    assert probed["mode"] == "probe" and probed["auroc"]["mortality_365d"] == 0.70
    assert tuned["mode"] == "finetune" and tuned["auroc"]["mortality_365d"] == 0.75
    assert probed["steps"] == tuned["steps"] == 10


def test_render_summary_shows_the_mode_column_and_the_ft_control() -> None:
    payload = {
        "grid": "g",
        "base": "configs/pretrain_default.yaml",
        "source": "physionet2019",
        "updated": "2026-09-10T00:00:00+00:00",
        "eval": {
            "bootstrap": 200,
            "eval_subject_limit": None,
            "eval_subject_seed": 0,
            "probe_features": "auto",
            "probe_layer": "final",
            "eval_modes": ["probe", "finetune"],
            "ft_epochs": 5,
        },
        "runs": [
            {"run": "ar_bins_s1", "mode": "probe", "auroc": {"sepsis_6h": 0.71}, "final": {}},
            {"run": "ar_bins_s1", "mode": "finetune", "auroc": {"sepsis_6h": 0.80}, "final": {}},
            # A legacy row, written before the mode column existed.
            {"run": "ar_cont_s1", "auroc": {"sepsis_6h": 0.70}, "final": {}},
        ],
        "baselines": {
            "gbm": {"sepsis_6h": 0.836},
            "ft_random@ar_bins_s1": {"sepsis_6h": 0.78},
        },
    }
    text = ablate.render_summary(payload)
    assert "| mode |" in text
    assert text.count("| `ar_bins_s1` | probe |") == 1
    assert text.count("| `ar_bins_s1` | finetune |") == 1
    assert "| `ar_cont_s1` | probe |" in text
    assert "Evaluation modes: `probe`, `finetune`" in text
    assert "`ft_random@<run>`" in text
    assert "None" not in text


def test_the_shipped_finetune_grids_plan_without_their_checkpoints(tmp_path: Path) -> None:
    """The a2ft grids are authored for the GPU host and must dry-run anywhere.

    Every cell reuses an ``a2-*`` checkpoint this machine may never have seen,
    so planning has to name the row (``REUSE?``) rather than raise; only
    ``run_grid`` insists the file is there.
    """
    for name in ("a2ft_physionet2019", "a2ft_physionet2012"):
        grid = ablate.load_grid(REPO / "configs" / "grids" / f"{name}.yaml")
        assert grid.eval_modes == ("finetune",), name
        assert grid.ft_epochs == 5 and grid.ft_balanced is False
        entries = ablate.plan(grid)
        assert [e["run"] for e in entries] == [
            "ar_bins_s1",
            "hybrid_bins_s1",
            "latent_cont_s1",
            "latent_only_s1",
            "hybrid_bins_lm_s1",
        ], name
        assert all(e["reuse"] for e in entries), "no cell may train"
        assert all(e["tokens"] == 0 for e in entries)
        for entry in entries:
            rows = [m for m in entry["eval_models"] if not m.startswith("ft_random")]
            assert rows == [f"ft:{entry['checkpoint']}"], entry["run"]
        # The LM cell fine-tunes at a smaller batch; every other cell at 64.
        batches = {e["run"]: e["ft_batch"] for e in entries}
        assert batches.pop("hybrid_bins_lm_s1") == 4
        assert set(batches.values()) == {64}
        # The two architectures that get a from-scratch control, and only those.
        with_controls = [e["run"] for e in entries if "ft_random" in e["eval_models"]]
        assert with_controls == ["ar_bins_s1", "hybrid_bins_lm_s1"]
        assert ablate.main([str(REPO / "configs" / "grids" / f"{name}.yaml"), "--dry-run"]) == 0


def test_a_cell_may_override_the_finetuning_batch(tmp_path: Path) -> None:
    """Per-cell, like ``budget_tokens``: a 0.5B encoder does not hold 64 windows."""
    path = _grid_file(
        tmp_path,
        eval_modes=["finetune"],
        ft_batch=32,
        runs=[
            {"name": "small", "overrides": {"objective.kind": "ar", "model.causal": True}},
            {
                "name": "lm",
                "overrides": {"objective.kind": "ar", "model.causal": True},
                "ft_batch": 4,
            },
        ],
    )
    grid = ablate.load_grid(path)
    assert grid.ft_batch == 32
    assert [e["ft_batch"] for e in ablate.plan(grid)] == [32, 4]

    # An unknown per-run key is still an error.
    with pytest.raises(ValueError, match="unknown keys"):
        ablate.load_grid(
            _grid_file(tmp_path, runs=[{"name": "x", "ft_epochs": 2}]),
        )


def test_training_cache_follows_the_grid_source(tmp_path, monkeypatch):
    """A cell must pretrain on the same source it is evaluated on."""
    import scripts.ablate as ab

    grid = ab.load_grid("configs/grids/a2_physionet2019.yaml")
    assert grid.train_cache_dir() == "data/cache/physionet2019"
    seen: list[list[str]] = []
    monkeypatch.setattr(ab, "_spawn", lambda cmd, log: seen.append([str(c) for c in cmd]))
    monkeypatch.setattr(ab, "_final_metrics", lambda out_dir: {"loss": 0.0})
    entry = ab.plan(grid)[0]
    entry = dict(entry, out_dir=str(tmp_path / "cell"))
    ab.train_one(grid, entry, ab.Log(tmp_path / "log.txt"))
    assert any(a == "data.cache_dir=data/cache/physionet2019" for a in seen[0])
