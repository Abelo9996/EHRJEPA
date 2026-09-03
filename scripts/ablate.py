"""``python scripts/ablate.py configs/grids/<grid>.yaml`` -- train, evaluate, tabulate.

One grid file is one experiment: a base config, a token budget, and a list of
named runs that each differ from the base by a handful of overrides. For every
run, in order, this trains to the budget, then hands the final checkpoint to the
phase-4 evaluation harness on the *same* held-out subject subset, bootstrap count
and seeds every other row was scored on, and appends one line to
``docs/experiments/<grid>/summary.md`` and ``summary.json``.

Three properties are what make the script worth having rather than a shell loop:

**Matched compute, not matched steps.** ``steps = ceil(budget_tokens / (batch x
max_len))``. A run that shortens its window or its batch gets proportionally more
steps, so "same budget" survives a config change instead of quietly becoming
"same number of gradient updates".

**Resumable, at run granularity.** A run whose row is already in
``summary.json`` is skipped. A 16 GB laptop running a six-run grid overnight will
be interrupted -- by a sleep, an OOM, a closed lid -- and the recovery has to be
"launch the same command again", not "work out which four runs finished".

**Baselines computed once.** ``lr`` and ``gbm`` are count-feature models with no
dependence on the encoder, so their held-out scores are read straight out of an
earlier run's ``predictions.parquet`` rather than refit six times.
``random_init`` *does* depend on the architecture, so it is computed once against
the grid's own first checkpoint and cached in ``baselines.json``: the control for
a 4x192 encoder has to be an untrained 4x192 encoder.

Detached use, which is the intended one::

    nohup python scripts/ablate.py configs/grids/pilot_desynpuf.yaml &
    tail -f runs/pilot_desynpuf/ablate.log
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO / "src"))

from ehrjepa.train.config import load_config  # noqa: E402

__all__ = ["Grid", "GridRun", "load_grid", "plan", "render_summary", "run_grid"]

#: Columns of the per-run table, before the per-task AUROC columns.
SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run", "run"),
    ("objective", "objective"),
    ("target_mode", "target"),
    ("lambda_sigreg", "lambda"),
    ("p_future", "p_future"),
    ("steps", "steps"),
    ("tokens", "tokens"),
    ("loss", "loss"),
    ("pred_loss", "pred"),
    ("ce", "ce"),
    ("top1", "top1"),
    ("top10", "top10"),
    ("effective_rank", "rank"),
    ("cos_gap", "cos_gap"),
    ("tokens_per_s", "tok/s"),
    ("wall_s", "wall_s"),
)


@dataclass
class GridRun:
    """One named cell of the grid."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    budget_tokens: int = 0

    def override_strings(self) -> list[str]:
        return [f"{key}={_scalar(value)}" for key, value in self.overrides.items()]


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass
class Grid:
    """A parsed grid file."""

    name: str
    base: Path
    runs: list[GridRun]
    source: str = "desynpuf-s1"
    budget_tokens: int = 12_000_000
    seed: int = 0
    bootstrap: int = 200
    eval_subject_limit: int | None = 3000
    eval_subject_seed: int = 0
    probe_features: str = "cls_mean"
    probe_layer: str = "final"
    tasks: str = "all"
    reuse_predictions: str | None = None
    reuse_models: tuple[str, ...] = ("lr", "gbm")
    control_models: tuple[str, ...] = ("random_init",)
    docs_root: Path = Path("docs/experiments")
    runs_root: Path = Path("runs")

    @property
    def doc_dir(self) -> Path:
        return REPO / self.docs_root / self.name

    @property
    def run_root(self) -> Path:
        return REPO / self.runs_root / self.name

    @property
    def summary_json(self) -> Path:
        return self.doc_dir / "summary.json"

    @property
    def summary_md(self) -> Path:
        return self.doc_dir / "summary.md"

    @property
    def log_path(self) -> Path:
        return self.run_root / "ablate.log"


def load_grid(path: str | Path) -> Grid:
    """Read a grid YAML. Unknown top-level keys are an error, not a shrug."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    known = {
        "name",
        "base",
        "runs",
        "source",
        "budget_tokens",
        "seed",
        "bootstrap",
        "eval_subject_limit",
        "eval_subject_seed",
        "probe_features",
        "probe_layer",
        "tasks",
        "reuse_predictions",
        "reuse_models",
        "control_models",
        "docs_root",
        "runs_root",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown grid keys: {sorted(unknown)}")
    if not raw.get("runs"):
        raise ValueError(f"{path} lists no runs")

    default_budget = int(raw.get("budget_tokens", 12_000_000))
    runs: list[GridRun] = []
    seen: set[str] = set()
    for item in raw["runs"]:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError(f"each run needs a name, got {item!r}")
        extra = set(item) - {"name", "overrides", "budget_tokens"}
        if extra:
            raise ValueError(f"run {item['name']!r} has unknown keys: {sorted(extra)}")
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"duplicate run name {name!r}")
        seen.add(name)
        runs.append(
            GridRun(
                name=name,
                overrides=dict(item.get("overrides") or {}),
                budget_tokens=int(item.get("budget_tokens", default_budget)),
            )
        )
    fields = {
        key: raw[key]
        for key in known - {"runs", "name", "base", "reuse_models", "control_models"}
        if key in raw
    }
    for key in ("docs_root", "runs_root"):
        if key in fields:
            fields[key] = Path(fields[key])
    for key in ("reuse_models", "control_models"):
        if key in raw:
            fields[key] = tuple(raw[key])
    return Grid(
        name=str(raw.get("name") or path.stem),
        base=Path(raw["base"]),
        runs=runs,
        **fields,
    )


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def steps_for(budget_tokens: int, batch_size: int, max_len: int) -> int:
    """Gradient steps that spend ``budget_tokens`` at ``batch_size x max_len``.

    The product is the *nominal* window: real batches are right-padded, so a step
    consumes this many token slots and somewhat fewer real events. Budgeting on
    the nominal number is what keeps two configs comparable -- the padded fraction
    is a property of the data and the window length, not of the objective.
    """
    per_step = batch_size * max_len
    if per_step <= 0:
        raise ValueError("batch_size and max_len must be positive")
    return max(1, math.ceil(budget_tokens / per_step))


def plan(grid: Grid) -> list[dict]:
    """One dict per run: resolved steps, directories, and whether it is done."""
    done = {row["run"] for row in _load_rows(grid.summary_json)}
    out = []
    for item in grid.runs:
        config = load_config(REPO / grid.base, item.override_strings())
        steps = steps_for(item.budget_tokens, config.run.batch_size, config.data.max_len)
        out.append(
            {
                "run": item.name,
                "overrides": dict(item.overrides),
                "budget_tokens": item.budget_tokens,
                "batch_size": config.run.batch_size,
                "max_len": config.data.max_len,
                "steps": steps,
                "tokens": steps * config.run.batch_size * config.data.max_len,
                "objective": config.objective.kind,
                # The three JEPA knobs are meaningless for an AR cell: it has no
                # target network, no SIGReg term and no context/target masking.
                # Reporting the base config's values there would invite a reader
                # to compare a column that does not exist in that run.
                "target_mode": _jepa_only(config, str(config.model.get("target_mode", "shared"))),
                "lambda_sigreg": _jepa_only(config, config.objective.lambda_sigreg),
                "p_future": _jepa_only(config, config.masking.p_future),
                "out_dir": str(grid.run_root / item.name),
                "eval_dir": str(grid.doc_dir / "eval" / item.name),
                "done": item.name in done,
            }
        )
    return out


def _short(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (tests point elsewhere)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _jepa_only(config: Any, value: Any) -> Any:
    return None if config.objective.kind == "ar" else value


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return list(payload.get("runs", []))


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


class Log:
    """Write to ``runs/<grid>/ablate.log`` and to stdout, unbuffered."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", buffering=1)

    def say(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.handle.write(line + "\n")
        print(line, flush=True)

    def close(self) -> None:
        self.handle.close()


def _spawn(command: Sequence[str], log: Log) -> None:
    log.say("$ " + " ".join(str(c) for c in command))
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(command, cwd=REPO, stdout=log.handle, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}: {command[:4]}")


def train_one(grid: Grid, entry: Mapping[str, Any], log: Log) -> dict:
    """Train one cell to its budget and return its last logged metrics row."""
    out_dir = Path(entry["out_dir"])
    overrides = [f"{k}={_scalar(v)}" for k, v in entry["overrides"].items()]
    started = time.perf_counter()
    _spawn(
        [
            sys.executable,
            "-m",
            "ehrjepa.train.pretrain",
            "--config",
            str(grid.base),
            "--override",
            f"run.steps={entry['steps']}",
            f"run.seed={grid.seed}",
            f"run.out_dir={out_dir}",
            "run.ckpt_every=0",
            *overrides,
        ],
        log,
    )
    wall = time.perf_counter() - started
    rows = list(csv.DictReader((out_dir / "metrics.csv").open()))
    final = {k: _number(v) for k, v in rows[-1].items()} if rows else {}
    steady = [_number(r["tokens_per_s"]) for r in rows[1:]] or [final.get("tokens_per_s", 0.0)]
    final["tokens_per_s"] = sum(steady) / len(steady)
    final["wall_s"] = round(wall, 1)
    return final


def _number(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def eval_one(grid: Grid, entry: Mapping[str, Any], with_controls: bool, log: Log) -> dict:
    """Score one checkpoint on the shared cohort; return ``{model: {task: auroc}}``."""
    checkpoint = Path(entry["out_dir"]) / "final.pt"
    models = [f"ckpt:{checkpoint}"]
    if with_controls:
        models = [*grid.control_models, *models]
    eval_dir = Path(entry["eval_dir"])
    command = [
        sys.executable,
        "-m",
        "ehrjepa.eval.run",
        "--source",
        grid.source,
        "--tasks",
        grid.tasks,
        "--models",
        ",".join(models),
        "--out",
        str(eval_dir),
        "--bootstrap",
        str(grid.bootstrap),
        "--seed",
        str(grid.seed),
        "--no-few-shot",
        "--probe-features",
        grid.probe_features,
        "--probe-layer",
        grid.probe_layer,
    ]
    if grid.eval_subject_limit:
        command += [
            "--eval-subject-limit",
            str(grid.eval_subject_limit),
            "--eval-subject-seed",
            str(grid.eval_subject_seed),
        ]
    _spawn(command, log)
    results = json.loads((eval_dir / "results.json").read_text())
    out: dict[str, dict[str, float]] = {}
    for task, body in results["tasks"].items():
        for model, record in body.get("models", {}).items():
            out.setdefault(model, {})[task] = record["metrics"]["auroc"]["point"]
    return out


def baselines_for(grid: Grid, fresh: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Reference AUROCs: reused count baselines plus any freshly computed control.

    Cached in ``baselines.json`` so a resumed grid does not recompute the control
    against a different checkpoint's architecture than the first pass used.
    """
    path = grid.doc_dir / "baselines.json"
    stored = json.loads(path.read_text()) if path.exists() else {}
    if not stored and grid.reuse_predictions:
        stored.update(_auroc_from_predictions(REPO / grid.reuse_predictions, grid.reuse_models))
    if fresh:
        for model in grid.control_models:
            if model in fresh:
                stored[model] = dict(fresh[model])
    if stored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    return stored


def _auroc_from_predictions(path: Path, models: Sequence[str]) -> dict:
    """Per-task AUROC for ``models``, recomputed from stored held-out scores."""
    if not path.exists():
        return {}
    import polars as pl

    from ehrjepa.eval.metrics import auroc

    frame = pl.read_parquet(path)
    out: dict[str, dict[str, float]] = {}
    for (model, task), part in frame.group_by(["model", "task"], maintain_order=True):
        if model not in models:
            continue
        y = part["label"].to_numpy()
        out.setdefault(str(model), {})[str(task)] = float(auroc(y, part["score"].to_numpy()))
    return out


def run_grid(grid: Grid, only: Sequence[str] | None = None, force: bool = False) -> dict:
    """Train and evaluate every unfinished cell, appending to the summary as it goes."""
    log = Log(grid.log_path)
    grid.doc_dir.mkdir(parents=True, exist_ok=True)
    entries = plan(grid)
    if only:
        entries = [e for e in entries if e["run"] in set(only)]
    payload = _read_summary(grid)
    log.say(
        f"grid {grid.name}: {len(entries)} run(s), "
        f"{sum(not e['done'] for e in entries)} outstanding, base {grid.base}"
    )
    try:
        for entry in entries:
            if entry["done"] and not force:
                log.say(f"skip {entry['run']} -- already in summary.json")
                continue
            log.say(
                f"start {entry['run']}: {entry['steps']} steps x "
                f"{entry['batch_size']}x{entry['max_len']} = {entry['tokens']:,} tokens"
            )
            final = train_one(grid, entry, log)
            log.say(f"trained {entry['run']} in {final['wall_s']:.0f}s")
            controls_needed = not (grid.doc_dir / "baselines.json").exists()
            scored = eval_one(grid, entry, controls_needed, log)
            baselines_for(grid, scored)
            row = _row(entry, final, scored)
            payload["runs"] = [r for r in payload["runs"] if r["run"] != row["run"]] + [row]
            payload["baselines"] = baselines_for(grid)
            _write_summary(grid, payload)
            log.say(f"done {entry['run']}: " + _auroc_line(row))
    finally:
        payload["baselines"] = baselines_for(grid)
        _write_summary(grid, payload)
        log.say("ablation finished")
        log.close()
    return payload


def _row(entry: Mapping[str, Any], final: Mapping[str, float], scored: Mapping) -> dict:
    name = entry["run"]
    matches = [k for k in scored if k.startswith("ckpt:")]
    auroc = dict(scored[matches[0]]) if matches else {}
    return {
        "run": name,
        "objective": entry["objective"],
        "target_mode": entry["target_mode"],
        "lambda_sigreg": entry["lambda_sigreg"],
        "p_future": entry["p_future"],
        "steps": entry["steps"],
        "tokens": entry["tokens"],
        "batch_size": entry["batch_size"],
        "max_len": entry["max_len"],
        "overrides": dict(entry["overrides"]),
        "final": {k: v for k, v in final.items() if isinstance(v, float)},
        "auroc": auroc,
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def _auroc_line(row: Mapping[str, Any]) -> str:
    return " ".join(f"{task}={value:.3f}" for task, value in sorted(row["auroc"].items()))


def _read_summary(grid: Grid) -> dict:
    if grid.summary_json.exists():
        payload = json.loads(grid.summary_json.read_text())
        payload.setdefault("runs", [])
        return payload
    return {"grid": grid.name, "base": str(grid.base), "runs": [], "baselines": {}}


def _write_summary(grid: Grid, payload: dict) -> None:
    payload["grid"] = grid.name
    payload["base"] = str(grid.base)
    payload["source"] = grid.source
    payload["updated"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    payload["eval"] = {
        "bootstrap": grid.bootstrap,
        "eval_subject_limit": grid.eval_subject_limit,
        "eval_subject_seed": grid.eval_subject_seed,
        "seed": grid.seed,
        "probe_features": grid.probe_features,
        "probe_layer": grid.probe_layer,
    }
    order = [item.name for item in grid.runs]
    payload["runs"].sort(key=lambda r: order.index(r["run"]) if r["run"] in order else 99)
    grid.doc_dir.mkdir(parents=True, exist_ok=True)
    grid.summary_json.write_text(json.dumps(payload, indent=2) + "\n")
    grid.summary_md.write_text(render_summary(payload))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if value != value:  # NaN
            return "--"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_summary(payload: Mapping[str, Any]) -> str:
    """The markdown table. Numbers only -- no ranking, no adjective, no verdict."""
    runs = list(payload.get("runs", []))
    baselines = dict(payload.get("baselines", {}))
    tasks: list[str] = []
    for row in runs:
        for task in row.get("auroc", {}):
            if task not in tasks:
                tasks.append(task)
    for model in baselines.values():
        for task in model:
            if task not in tasks:
                tasks.append(task)
    tasks.sort()

    lines = [f"# Ablation grid -- {payload.get('grid', '?')}", ""]
    meta = payload.get("eval", {})
    lines += [
        f"Base config `{payload.get('base', '')}`, source `{payload.get('source', '')}`, "
        f"held-out AUROC on a {meta.get('eval_subject_limit')}-subject subset "
        f"(seed {meta.get('eval_subject_seed')}), {meta.get('bootstrap')} bootstrap resamples, "
        f"probe `{meta.get('probe_features')}@{meta.get('probe_layer')}`.",
        "",
        f"Rows are appended by `scripts/ablate.py` as each run finishes. Last update: "
        f"{payload.get('updated', '')}.",
        "",
    ]

    header = [label for _, label in SUMMARY_COLUMNS] + tasks
    body = []
    for row in runs:
        final = row.get("final", {})
        cells = []
        for key, _ in SUMMARY_COLUMNS:
            if key == "run":
                cells.append(f"`{row['run']}`")
            elif key in row:
                cells.append(_fmt(row[key]))
            else:
                cells.append(_fmt(final.get(key)))
        cells += [_fmt(row.get("auroc", {}).get(task)) for task in tasks]
        body.append(cells)
    lines += _table(header, body)

    if baselines:
        lines += ["", "## Reference models", ""]
        lines += _table(
            ["model", *tasks],
            [
                [f"`{name}`"] + [_fmt(scores.get(task)) for task in tasks]
                for name, scores in sorted(baselines.items())
            ],
        )
        lines += [
            "",
            "`lr` and `gbm` are count-feature baselines and do not depend on the encoder, so "
            "their scores are reused from an earlier run's `predictions.parquet`. "
            "`random_init` is this grid's own architecture with untrained weights.",
        ]
    return "\n".join(lines) + "\n"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument("--only", default=None, help="comma-separated run names")
    parser.add_argument("--force", action="store_true", help="rerun cells already summarised")
    args = parser.parse_args(list(argv) if argv is not None else None)

    grid = load_grid(args.grid)
    only = args.only.split(",") if args.only else None
    if args.dry_run:
        entries = plan(grid)
        if only:
            entries = [e for e in entries if e["run"] in set(only)]
        print(f"grid {grid.name}: base {grid.base}, source {grid.source}")
        print(f"summary {_short(grid.summary_json)}  log {_short(grid.log_path)}")
        total = 0
        for entry in entries:
            state = "SKIP (done)" if entry["done"] and not args.force else "RUN"
            total += 0 if entry["done"] and not args.force else entry["tokens"]
            print(
                f"  {state:<11} {entry['run']:<18} {entry['steps']:>6} steps x "
                f"{entry['batch_size']}x{entry['max_len']} = {entry['tokens']:>12,} tokens  "
                f"{entry['objective']}/{_fmt(entry['target_mode'])} "
                f"lambda={_fmt(entry['lambda_sigreg'])} p_future={_fmt(entry['p_future'])}"
            )
        print(f"  total outstanding: {total:,} tokens")
        return 0
    run_grid(grid, only=only, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
