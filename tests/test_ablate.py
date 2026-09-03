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


def test_a_per_run_budget_overrides_the_grid_default(tmp_path: Path) -> None:
    path = _grid_file(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["runs"][0]["budget_tokens"] = 1_638_400
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    entries = ablate.plan(ablate.load_grid(path))
    assert entries[0]["steps"] == 100
    assert entries[1]["steps"] == 200


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
    header, rule = text.splitlines()[6], text.splitlines()[7]
    assert header.count("|") == rule.count("|")
