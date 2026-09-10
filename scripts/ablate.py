"""``python scripts/ablate.py configs/grids/<grid>.yaml`` -- train, evaluate, tabulate.

One grid file is one experiment: a base config, a token budget, and a list of
named runs that each differ from the base by a handful of overrides. For every
run, in order, this trains to the budget, then hands the final checkpoint to the
phase-4 evaluation harness on the *same* held-out subject subset, bootstrap count
and seeds every other row was scored on, and appends one line to
``docs/experiments/<grid>/summary.md`` and ``summary.json``.

Three properties are what make the script worth having rather than a shell loop:

**Matched compute, not matched steps.** ``steps = ceil(budget_tokens / (batch x
max_len x accum_steps))``. A run that shortens its window or its batch gets
proportionally more steps, so "same budget" survives a config change instead of
quietly becoming "same number of gradient updates" -- and a run that trades
batch size for gradient accumulation to fit a VRAM budget (``batch_size: 32``,
``optim.accum_steps: 2`` in place of ``batch_size: 64``) spends the *same*
nominal tokens per step it did before the trade, because the trainer's ``step``
counter advances once per optimizer step, not once per micro-batch.

**Resumable, at run granularity and mid-run.** A run whose row is already in
``summary.json`` is skipped. A cell still training when a grid is interrupted --
by a sleep, an OOM, a closed lid, a ``systemctl stop`` -- picks up from its own
``latest.pt`` rather than restarting from step zero: the trainer checkpoints
periodically during a cell (``grid.ckpt_every``, default every 2000 steps) and
the next launch of the same command hands it ``run.resume=<out_dir>/latest.pt``
whenever that file exists and the cell has not yet produced a ``final.pt``. A 16
GB laptop running a six-run grid overnight will be interrupted, and the recovery
has to be "launch the same command again", not "work out which four runs
finished and how far the fifth one got".

**Re-evaluation without retraining.** A run may name ``reuse_checkpoint:``
instead of training: the cell is scored from a checkpoint an earlier grid already
produced, its metrics read out of that run's ``metrics.csv``, and its row lands in
*this* grid's summary. That is what makes "the same checkpoint under a different
pooling" a row rather than a footnote, and it never writes into the other grid's
directory.

**Probe or fine-tune, or both.** ``eval_modes: [probe, finetune]`` scores each
cell twice: a logistic probe on the frozen encoder, and the same encoder trained
end to end on the task the way the ICU literature compares an encoder to
gradient boosting (:mod:`ehrjepa.eval.finetune`). Each mode is one row of the
summary, told apart by the ``mode`` column, and both ride in the same eval
command so the two are scored on identical anchors. The default is ``[probe]``
alone, so every grid written before this existed plans and emits exactly the
rows it did before.

**Baselines computed once.** ``lr`` and ``gbm`` are count-feature models with no
dependence on the encoder, so their held-out scores are read straight out of an
earlier run's ``predictions.parquet`` (``reuse_predictions:``) rather than refit
six times. A grid with no ``reuse_predictions`` -- because the earlier run's
cohort does not match this grid's, e.g. a full-held-out grid cannot reuse a
3,000-subject run's scores -- fits them itself instead, once, riding along on
whichever cell's eval command runs first; every later cell finds them already in
``baselines.json`` and asks for nothing more.
``random_init`` *does* depend on the architecture, so it is computed once per
cell named in ``control_runs`` and cached in ``baselines.json`` as
``random_init@<run>``: the control for a 4x192 encoder has to be an untrained
4x192 encoder, and the control for a *causal* one has to be causal, because an
untrained causal encoder's CLS row is a constant and its probe is a different
probe.

Detached use, which is the intended one::

    nohup python scripts/ablate.py configs/grids/pilot_desynpuf.yaml &
    tail -f runs/2026-09-03-pilot-desynpuf/ablate.log
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

#: How each evaluation mode names a checkpoint's row and its untrained control
#: in ``ehrjepa.eval.run --models``. ``probe`` fits a logistic regression on
#: frozen features; ``finetune`` trains the encoder end to end on the task, the
#: way the ICU literature compares an encoder to gradient boosting. A grid lists
#: the modes it wants in ``eval_modes:`` and gets one summary row per (cell,
#: mode); the default is ``[probe]``, so every grid written before this existed
#: emits exactly the rows it did before.
EVAL_MODES: dict[str, tuple[str, str]] = {
    "probe": ("ckpt:", "random_init"),
    "finetune": ("ft:", "ft_random"),
}

#: Columns of the per-run table, before the per-task AUROC columns.
SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run", "run"),
    ("mode", "mode"),
    ("objective", "objective"),
    ("pooling", "probe"),
    # Blank outside ``finetune`` mode. Present because a cell may lower it (an
    # LM encoder does not hold 64 windows), and a fine-tuned AUROC that was
    # produced at a different batch has to say so.
    ("ft_batch", "ft_bs"),
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
    """One named cell of the grid.

    ``reuse_checkpoint`` is a repo-relative path to an existing ``*.pt``. A cell
    that names one is not trained: it is evaluated from that file, and its
    training metrics are read from the ``metrics.csv`` beside it.
    """

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    budget_tokens: int = 0
    reuse_checkpoint: str | None = None
    #: ``finetune`` mode only: this cell's fine-tuning batch, overriding the
    #: grid's. Per-cell for the same reason ``budget_tokens`` is: a cell whose
    #: encoder is a 0.5B language model does not hold the 64 windows a 13M one
    #: does, and the alternative is shrinking every cell's batch to fit the
    #: largest model in the grid.
    ft_batch: int | None = None

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
    #: ``auto`` resolves per checkpoint inside the harness (``last`` for causal,
    #: ``mean`` for bidirectional); the resolved value lands in each row's
    #: ``pooling``.
    probe_features: str = "auto"
    probe_layer: str = "final"
    tasks: str = "all"
    reuse_predictions: str | None = None
    reuse_models: tuple[str, ...] = ("lr", "gbm")
    control_models: tuple[str, ...] = ("random_init",)
    #: Cells whose architecture gets its own untrained control. Empty means "the
    #: first run". A causal encoder and a bidirectional one are different models
    #: even untrained -- the causal one's CLS row is a constant -- so a grid that
    #: mixes objectives wants one control per objective, not one per grid.
    control_runs: tuple[str, ...] = ()
    #: Which evaluation modes every cell gets, in :data:`EVAL_MODES`. The
    #: default is the frozen probe alone, which is what every grid in
    #: ``configs/grids/`` predating fine-tuning was run under.
    eval_modes: tuple[str, ...] = ("probe",)
    #: ``finetune`` mode only: the fine-tuning epoch cap and whether each epoch
    #: is drawn class-balanced. Passed through to ``ehrjepa.eval.run``, and
    #: absent from the command entirely when no mode asks for it.
    ft_epochs: int = 5
    ft_balanced: bool = False
    #: Windows per fine-tuning step, overridable per cell (``GridRun.ft_batch``).
    ft_batch: int = 64
    docs_root: Path = Path("docs/experiments")
    runs_root: Path = Path("runs")
    #: How often a training cell checkpoints to ``latest.pt``. Non-zero so a cell
    #: interrupted mid-run can resume instead of restarting from step zero.
    ckpt_every: int = 2000

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
        "control_runs",
        "eval_modes",
        "ft_epochs",
        "ft_balanced",
        "ft_batch",
        "docs_root",
        "runs_root",
        "ckpt_every",
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
        extra = set(item) - {
            "name",
            "overrides",
            "budget_tokens",
            "reuse_checkpoint",
            "ft_batch",
        }
        if extra:
            raise ValueError(f"run {item['name']!r} has unknown keys: {sorted(extra)}")
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"duplicate run name {name!r}")
        seen.add(name)
        reuse = item.get("reuse_checkpoint")
        if reuse and item.get("overrides"):
            raise ValueError(f"run {name!r} reuses a checkpoint, so its overrides do nothing")
        runs.append(
            GridRun(
                name=name,
                overrides=dict(item.get("overrides") or {}),
                budget_tokens=int(item.get("budget_tokens", default_budget)),
                reuse_checkpoint=str(reuse) if reuse else None,
                ft_batch=int(item["ft_batch"]) if item.get("ft_batch") else None,
            )
        )
    fields = {
        key: raw[key]
        for key in known
        - {"runs", "name", "base", "reuse_models", "control_models", "control_runs", "eval_modes"}
        if key in raw
    }
    for key in ("docs_root", "runs_root"):
        if key in fields:
            fields[key] = Path(fields[key])
    for key in ("reuse_models", "control_models", "control_runs", "eval_modes"):
        if key in raw:
            fields[key] = tuple(raw[key])
    fields.setdefault("control_runs", (runs[0].name,))
    fields.setdefault("eval_modes", ("probe",))
    unknown_modes = set(fields["eval_modes"]) - set(EVAL_MODES)
    if not fields["eval_modes"] or unknown_modes:
        raise ValueError(f"eval_modes must be a non-empty subset of {sorted(EVAL_MODES)}")
    unknown_control = set(fields["control_runs"]) - seen
    if unknown_control:
        raise ValueError(f"control_runs names no such run: {sorted(unknown_control)}")
    return Grid(
        name=str(raw.get("name") or path.stem),
        base=Path(raw["base"]),
        runs=runs,
        **fields,
    )


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def steps_for(budget_tokens: int, batch_size: int, max_len: int, accum_steps: int = 1) -> int:
    """*Optimizer* steps that spend ``budget_tokens`` at ``batch_size x max_len``.

    The product is the *nominal* window: real batches are right-padded, so a step
    consumes this many token slots and somewhat fewer real events. Budgeting on
    the nominal number is what keeps two configs comparable -- the padded fraction
    is a property of the data and the window length, not of the objective.

    ``accum_steps`` has to be in this product, not just ``batch_size x max_len``:
    the trainer's ``step`` counter (see ``Trainer.train``) advances once per
    *optimizer* step, and each optimizer step runs ``accum_steps`` micro-batches
    of ``batch_size x max_len`` before it calls ``optimizer.step()``. A cell that
    halves its batch size and sets ``accum_steps: 2`` to fit a fixed VRAM budget
    (same effective batch, half the activation memory) consumes the *same*
    nominal tokens per optimizer step as it did unaccumulated -- so its step
    count, and therefore its ``run.steps`` override, must not be computed as if
    ``accum_steps`` were still 1, or the cell would silently spend twice its
    budget.
    """
    per_step = batch_size * max_len * max(1, accum_steps)
    if per_step <= 0:
        raise ValueError("batch_size, max_len and accum_steps must be positive")
    return max(1, math.ceil(budget_tokens / per_step))


def plan(grid: Grid) -> list[dict]:
    """One dict per run: resolved steps, directories, and whether it is done."""
    done = _finished_runs(grid)
    out = []
    for item in grid.runs:
        entry = _reuse_entry(item) if item.reuse_checkpoint else _train_entry(grid, item)
        controls = item.name in grid.control_runs
        entry.update(
            run=item.name,
            overrides=dict(item.overrides),
            budget_tokens=item.budget_tokens,
            eval_dir=str(grid.doc_dir / "eval" / item.name),
            eval_models=eval_models(grid, Path(entry["checkpoint"]), controls),
            ft_batch=item.ft_batch or grid.ft_batch,
            done=item.name in done,
        )
        out.append(entry)
    return out


def _finished_runs(grid: Grid) -> set[str]:
    """Cells whose summary row is already present **for every mode** this grid runs.

    A grid that gains a mode therefore re-evaluates its cells for the missing
    mode instead of reporting them done; rows written before ``eval_modes``
    existed carry no ``mode`` and are read as the probe rows they are.
    """
    seen: dict[str, set[str]] = {}
    for row in _load_rows(grid.summary_json):
        seen.setdefault(str(row["run"]), set()).add(str(row.get("mode", "probe")))
    wanted = set(grid.eval_modes)
    return {name for name, modes in seen.items() if wanted <= modes}


def mode_controls(grid: Grid, mode: str) -> tuple[str, ...]:
    """``grid.control_models``, renamed for the row kind ``mode`` produces.

    The untrained control for a fine-tuned row is the same architecture
    *fine-tuned from scratch* (``ft_random``), not a frozen probe on untrained
    weights: a probe control cannot say whether the pretrained weights were a
    better initialisation than none.
    """
    control = EVAL_MODES[mode][1]
    return tuple(control if name == "random_init" else name for name in grid.control_models)


def control_specs(grid: Grid) -> list[str]:
    """Every control model spec this grid asks for, across its modes, once each."""
    out: list[str] = []
    for mode in grid.eval_modes:
        for name in mode_controls(grid, mode):
            if name not in out:
                out.append(name)
    return out


def eval_models(grid: Grid, checkpoint: Path, with_controls: bool) -> list[str]:
    """The ``--models`` list for one cell: one row per mode, controls first.

    Every mode rides in the *same* command, so a probe row and a fine-tune row
    of the same cell are scored on identical anchors and land in one
    ``predictions.parquet`` -- which is what makes the paired bootstrap between
    them a comparison rather than a coincidence.
    """
    models = [f"{EVAL_MODES[mode][0]}{checkpoint}" for mode in grid.eval_modes]
    return [*control_specs(grid), *models] if with_controls else models


def _train_entry(grid: Grid, item: GridRun) -> dict:
    """The cell resolved from the grid's base config plus this run's overrides."""
    config = load_config(REPO / grid.base, item.override_strings())
    accum = max(1, config.optim.accum_steps)
    steps = steps_for(item.budget_tokens, config.run.batch_size, config.data.max_len, accum)
    out_dir = grid.run_root / item.name
    return {
        "batch_size": config.run.batch_size,
        "max_len": config.data.max_len,
        "steps": steps,
        "tokens": steps * config.run.batch_size * config.data.max_len * accum,
        "objective": config.objective.kind,
        # The three JEPA knobs are meaningless for an AR cell: it has no target
        # network, no SIGReg term and no context/target masking. Reporting the
        # base config's values there would invite a reader to compare a column
        # that does not exist in that run.
        "target_mode": _latent_only(
            config.objective.kind, str(config.model.get("target_mode", "shared"))
        ),
        "lambda_sigreg": _latent_only(config.objective.kind, config.objective.lambda_sigreg),
        "p_future": _masked_only(config.objective.kind, config.masking.p_future),
        "out_dir": str(out_dir),
        "checkpoint": str(out_dir / "final.pt"),
        "reuse": False,
    }


def _reuse_entry(item: GridRun) -> dict:
    """The cell resolved from the *finished* run a ``reuse_checkpoint`` points into.

    Everything describing the training comes from that run's own ``config.json``,
    not from this grid's base: the row has to say what was actually trained, and
    this grid's base may differ from the one that trained it.

    A checkpoint that was trained elsewhere -- a GPU machine this grid was
    authored on but has not run on yet -- has no local ``config.json``. Planning
    still has to name the row, so this returns a placeholder rather than
    raising; ``run_grid`` fails loudly, naming the missing path, the moment it
    actually tries to evaluate the cell.
    """
    assert item.reuse_checkpoint is not None
    checkpoint = REPO / item.reuse_checkpoint
    source = checkpoint.parent
    config_path = source / "config.json"
    if not config_path.exists():
        return {
            "batch_size": 0,
            "max_len": 0,
            "steps": 0,
            "tokens": 0,
            "objective": "?",
            "target_mode": None,
            "lambda_sigreg": None,
            "p_future": None,
            "out_dir": str(source),
            "checkpoint": str(checkpoint),
            "reuse": True,
            "missing": True,
        }
    config = json.loads(config_path.read_text())
    kind = str(config["objective"]["kind"])
    batch_size = int(config["run"]["batch_size"])
    max_len = int(config["data"]["max_len"])
    steps = int(config["run"]["steps"])
    accum = max(1, int(config.get("optim", {}).get("accum_steps", 1)))

    return {
        "batch_size": batch_size,
        "max_len": max_len,
        "steps": steps,
        "tokens": steps * batch_size * max_len * accum,
        "objective": kind,
        "target_mode": _latent_only(kind, str(config["model"].get("target_mode", "shared"))),
        "lambda_sigreg": _latent_only(kind, config["objective"]["lambda_sigreg"]),
        "p_future": _masked_only(kind, config["masking"]["p_future"]),
        "out_dir": str(source),
        "checkpoint": str(checkpoint),
        "reuse": True,
    }


def _short(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (tests point elsewhere)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _latent_only(kind: str, value: Any) -> Any:
    """Blank for ``ar``, which has no target network and no SIGReg term."""
    return None if kind == "ar" else value


def _masked_only(kind: str, value: Any) -> Any:
    """Blank for everything but ``jepa``: only it samples context/target masks.

    The causal objectives take their context from the attention mask and their
    targets from a horizon, so reporting the base config's ``p_future`` on one of
    their rows would invite a reader to compare a column that does not exist in
    that run.
    """
    return value if kind == "jepa" else None


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


def train_one(grid: Grid, entry: Mapping[str, Any], log: Log, force: bool = False) -> dict:
    """Train one cell to its budget and return its last logged metrics row.

    If the cell already has a ``final.pt`` and ``metrics.csv`` (a previous grid
    attempt trained it but failed later, e.g. in eval), training is skipped and
    those artifacts are reused unless ``force`` is set. Otherwise, if the cell has
    a ``latest.pt`` but no ``final.pt`` -- it was mid-run when the grid was last
    interrupted -- the trainer is handed ``run.resume`` so it continues from that
    checkpoint's step instead of restarting at zero.
    """
    out_dir = Path(entry["out_dir"])
    if not force and (out_dir / "final.pt").exists() and (out_dir / "metrics.csv").exists():
        log.say(f"reusing trained checkpoint for {entry['run']} at {out_dir / 'final.pt'}")
        final = _final_metrics(out_dir)
        final["wall_s"] = 0.0
        return final
    resume_args: list[str] = []
    latest = out_dir / "latest.pt"
    if not force and latest.exists() and not (out_dir / "final.pt").exists():
        step = _checkpoint_step(latest)
        if step is not None:
            log.say(f"resuming {entry['run']} from step {step}")
        else:
            log.say(f"resuming {entry['run']} from {latest}")
        resume_args = [f"run.resume={latest}"]
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
            f"run.ckpt_every={grid.ckpt_every}",
            *resume_args,
            *overrides,
        ],
        log,
    )
    wall = time.perf_counter() - started
    final = _final_metrics(out_dir)
    final["wall_s"] = round(wall, 1)
    return final


def _checkpoint_step(path: Path) -> int | None:
    """The ``step`` recorded in a checkpoint, or ``None`` if it cannot be read.

    Used only for the log line: a corrupt or partially-written checkpoint (e.g.
    killed mid-``torch.save``) must not stop the resume, just make its log line
    less specific.
    """
    try:
        import torch

        return int(torch.load(path, map_location="cpu", weights_only=False)["step"])
    except Exception:
        return None


def _final_metrics(out_dir: Path) -> dict:
    """The last logged row of a run's ``metrics.csv``, with a steady-state tok/s."""
    rows = list(csv.DictReader((out_dir / "metrics.csv").open()))
    final = {k: _number(v) for k, v in rows[-1].items()} if rows else {}
    steady = [_number(r["tokens_per_s"]) for r in rows[1:]] or [final.get("tokens_per_s", 0.0)]
    final["tokens_per_s"] = sum(steady) / len(steady)
    return final


def reuse_one(entry: Mapping[str, Any], log: Log) -> dict:
    """No training: read the reused run's final metrics and report zero wall time."""
    log.say(f"reuse {entry['run']}: scoring {_short(Path(entry['checkpoint']))} as trained")
    final = _final_metrics(Path(entry["out_dir"]))
    final["wall_s"] = 0.0
    return final


def _number(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def eval_one(
    grid: Grid,
    entry: Mapping[str, Any],
    with_controls: bool,
    log: Log,
    with_baselines: bool = False,
) -> tuple[dict, dict]:
    """Score one checkpoint on the shared cohort.

    Returns ``({model: {task: auroc}}, {model: pooling})``. The pooling comes back
    from the harness rather than from the grid file because ``probe_features:
    auto`` is resolved per checkpoint, and a row that does not say which pooling
    produced its AUROCs is not comparable to anything.

    ``with_baselines`` adds ``grid.reuse_models`` (``lr``, ``gbm`` by default) to
    this cell's command. It is only set when the grid has no
    ``reuse_predictions`` file to read them from and they are not already cached
    in ``baselines.json`` -- so they are fit exactly once per grid, on whichever
    cell runs first, rather than once per row.

    One command carries every mode in ``grid.eval_modes``, so the returned
    mapping holds a ``ckpt:``-prefixed key per probed cell and an ``ft:``-prefixed
    one per fine-tuned cell; :func:`_row` picks the one its mode names.
    """
    checkpoint = Path(entry["checkpoint"])
    models = eval_models(grid, checkpoint, with_controls)
    if with_baselines:
        models = [*grid.reuse_models, *models]
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
    if "finetune" in grid.eval_modes:
        command += [
            "--ft-epochs",
            str(grid.ft_epochs),
            "--ft-batch",
            str(entry.get("ft_batch") or grid.ft_batch),
        ]
        if grid.ft_balanced:
            command.append("--ft-balanced")
    _spawn(command, log)
    results = json.loads((eval_dir / "results.json").read_text())
    out: dict[str, dict[str, float]] = {}
    for task, body in results["tasks"].items():
        for model, record in body.get("models", {}).items():
            out.setdefault(model, {})[task] = record["metrics"]["auroc"]["point"]
    pooling = {name: spec.get("features", "") for name, spec in results["models"].items()}
    return out, pooling


def control_name(model: str, run: str) -> str:
    """``random_init@jepa_ema`` -- which cell's architecture this control copies."""
    return f"{model}@{run}"


def baselines_needed(grid: Grid, have: Mapping[str, Any]) -> bool:
    """Whether ``lr``/``gbm`` still need fitting for this grid.

    True only when the grid has no ``reuse_predictions`` file to read them from
    *and* at least one of ``grid.reuse_models`` is missing from the cached
    baselines -- so they ride along on exactly one cell's eval command, first
    come first served, rather than being refit on every row.
    """
    return not grid.reuse_predictions and not all(m in have for m in grid.reuse_models)


def baselines_for(
    grid: Grid, fresh: Mapping[str, Mapping[str, float]] | None = None, run: str = ""
) -> dict:
    """Reference AUROCs: reused count baselines plus any freshly computed control.

    Cached in ``baselines.json`` so a resumed grid does not recompute a control
    against a different architecture than the first pass used, and so a control
    survives being asked for after the cell that produced it has finished. When
    the grid names no ``reuse_predictions`` file, ``lr``/``gbm`` are instead
    read out of ``fresh`` (whichever cell's command carried them, see
    :func:`baselines_needed`) and cached under their bare model name, exactly as
    :func:`_auroc_from_predictions` would have stored them -- so a row that asks
    "what did the count baselines score" cannot tell which path produced them.
    """
    path = grid.doc_dir / "baselines.json"
    stored = json.loads(path.read_text()) if path.exists() else {}
    if grid.reuse_predictions and not any(m in stored for m in grid.reuse_models):
        stored.update(_auroc_from_predictions(REPO / grid.reuse_predictions, grid.reuse_models))
    if fresh and run:
        for model in control_specs(grid):
            if model in fresh:
                stored[control_name(model, run)] = dict(fresh[model])
        if not grid.reuse_predictions:
            for model in grid.reuse_models:
                if model in fresh and model not in stored:
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
            if entry["reuse"]:
                final = reuse_one(entry, log)
            else:
                log.say(
                    f"start {entry['run']}: {entry['steps']} steps x "
                    f"{entry['batch_size']}x{entry['max_len']} = {entry['tokens']:,} tokens"
                )
                final = train_one(grid, entry, log, force=force)
                log.say(f"trained {entry['run']} in {final['wall_s']:.0f}s")
            have = baselines_for(grid)
            controls_needed = entry["run"] in grid.control_runs and not all(
                control_name(m, entry["run"]) in have for m in control_specs(grid)
            )
            scored, pooling = eval_one(
                grid, entry, controls_needed, log, baselines_needed(grid, have)
            )
            baselines_for(grid, scored, entry["run"])
            for mode in grid.eval_modes:
                row = _row(entry, final, scored, pooling, mode)
                payload["runs"] = [
                    r
                    for r in payload["runs"]
                    if (r["run"], r.get("mode", "probe")) != (row["run"], mode)
                ] + [row]
                log.say(f"done {entry['run']} [{mode}]: " + _auroc_line(row))
            payload["baselines"] = baselines_for(grid)
            _write_summary(grid, payload)
    finally:
        payload["baselines"] = baselines_for(grid)
        _write_summary(grid, payload)
        log.say("ablation finished")
        log.close()
    return payload


def _row(
    entry: Mapping[str, Any],
    final: Mapping[str, float],
    scored: Mapping,
    pooling: Mapping[str, str] | None = None,
    mode: str = "probe",
) -> dict:
    """One summary row: this cell as scored under one evaluation mode.

    The training columns are the cell's and are therefore identical across its
    modes; only ``mode``, ``pooling`` and the per-task AUROCs differ.
    """
    name = entry["run"]
    matches = [k for k in scored if k.startswith(EVAL_MODES[mode][0])]
    auroc = dict(scored[matches[0]]) if matches else {}
    return {
        "run": name,
        "mode": mode,
        "ft_batch": entry.get("ft_batch") if mode == "finetune" else None,
        "objective": entry["objective"],
        "pooling": (pooling or {}).get(matches[0]) if matches else None,
        "reuse_checkpoint": _short(Path(entry["checkpoint"])) if entry["reuse"] else None,
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
        "eval_modes": list(grid.eval_modes),
    }
    if "finetune" in grid.eval_modes:
        payload["eval"]["ft_epochs"] = grid.ft_epochs
        payload["eval"]["ft_balanced"] = grid.ft_balanced
        payload["eval"]["ft_batch"] = grid.ft_batch
    order = [item.name for item in grid.runs]
    modes = list(grid.eval_modes)

    def position(row: Mapping[str, Any]) -> tuple[int, int]:
        mode = str(row.get("mode", "probe"))
        return (
            order.index(row["run"]) if row["run"] in order else 99,
            modes.index(mode) if mode in modes else 99,
        )

    payload["runs"].sort(key=position)
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
    limit = meta.get("eval_subject_limit")
    cohort = (
        f"a {limit}-subject subset (seed {meta.get('eval_subject_seed')})"
        if limit
        else "the full held-out split"
    )
    modes = [str(m) for m in meta.get("eval_modes") or ["probe"]]
    mode_note = "" if modes == ["probe"] else f" Evaluation modes: `{'`, `'.join(modes)}`."
    lines += [
        f"Base config `{payload.get('base', '')}`, source `{payload.get('source', '')}`, "
        f"held-out AUROC on {cohort}, {meta.get('bootstrap')} bootstrap resamples, "
        f"probe `{meta.get('probe_features')}@{meta.get('probe_layer')}` "
        f"(the `probe` column gives each row's resolved pooling).{mode_note}",
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
            elif key == "mode":
                # Rows written before ``eval_modes`` existed carry no mode and
                # are the probe rows they say they are.
                cells.append(_fmt(row.get("mode", "probe")))
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
            "`random_init@<run>` is that run's own architecture with untrained weights, probed "
            "identically -- the control for a causal encoder is an untrained causal encoder.",
        ]
        if any(name.startswith("ft_random@") for name in baselines):
            lines += [
                "",
                "`ft_random@<run>` is that architecture *fine-tuned from untrained weights* on "
                "each task -- the train-from-scratch arm the `finetune` rows are read against.",
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
            skip = entry["done"] and not args.force
            if skip:
                state = "SKIP (done)"
            elif entry.get("missing"):
                state = "REUSE?"
            elif entry["reuse"]:
                state = "REUSE"
            else:
                state = "RUN"
            total += 0 if skip or entry["reuse"] else entry["tokens"]
            note = "  (checkpoint not found locally -- plan only)" if entry.get("missing") else ""
            print(
                f"  {state:<11} {entry['run']:<18} {entry['steps']:>6} steps x "
                f"{entry['batch_size']}x{entry['max_len']} = {entry['tokens']:>12,} tokens  "
                f"{entry['objective']}/{_fmt(entry['target_mode'])} "
                f"lambda={_fmt(entry['lambda_sigreg'])} p_future={_fmt(entry['p_future'])}{note}"
            )
            # The eval rows this cell will produce, one per mode plus whatever
            # controls it carries. Printed only when the grid asks for more than
            # the historical probe-only evaluation, so a default grid's plan is
            # byte-for-byte what it was.
            if tuple(grid.eval_modes) != ("probe",):
                for model in entry["eval_models"]:
                    kind, _, path = model.partition(":")
                    shown = f"{kind}:{_short(Path(path))}" if path else kind
                    print(f"    {'eval':<9} {shown}")
        print(f"  total outstanding: {total:,} tokens")
        return 0
    run_grid(grid, only=only, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
