"""``python -m ehrjepa.eval.run`` -- one command, one comparison table.

Every model in a run sees the *same* anchors: the task frame is built once, the
rows present in the tensor cache are kept once, and the train/tuning/held_out
partition of that frame is what both the count baselines and every encoder probe
are fit and scored on. That is what makes the paired bootstrap legitimate and
what stops a checkpoint from looking better because it was scored on an easier
subset.

::

    python -m ehrjepa.eval.run --source desynpuf-s1 --tasks all \\
        --models lr,gbm,random_init,ckpt:runs/sanity-A-default/final.pt \\
        --out docs/experiments/2026-09-03-eval-desynpuf/
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

from ehrjepa.eval import baselines, probe, report, tasks
from ehrjepa.eval.history import HistoryReader
from ehrjepa.eval.metrics import bootstrap_ci, paired_bootstrap

log = logging.getLogger(__name__)

SPLITS = ("train", "tuning", "held_out")


def restrict_eval_split(
    anchors: pl.DataFrame, eval_split: str, limit: int | None, seed: int
) -> pl.DataFrame:
    """Keep every ``train``/``tuning`` row, and at most ``limit`` subjects of ``eval_split``.

    The kept subjects are a seeded random draw, independent of task and of row
    order: each eval-split subject gets ``blake2b("eval_subset:<seed>:<id>")`` and
    the ``limit`` lowest hashes are kept, the same construction
    :func:`ehrjepa.eval.tasks.select_anchors` uses for the anchor draw. A subject
    kept for one task is therefore kept for every other task where it appears,
    without the tasks needing to share an anchor set.
    """
    if limit is None:
        return anchors
    eval_rows = anchors.filter(pl.col("split") == eval_split)
    if eval_rows.height <= limit:
        return anchors
    subjects = eval_rows["subject_id"].unique().sort().to_list()
    draws = pl.DataFrame(
        {
            "subject_id": subjects,
            "_draw": [
                int.from_bytes(
                    hashlib.blake2b(f"eval_subset:{seed}:{s}".encode(), digest_size=8).digest(),
                    "big",
                )
                for s in subjects
            ],
        },
        schema={"subject_id": anchors.schema["subject_id"], "_draw": pl.UInt64},
    )
    keep = draws.sort("_draw", "subject_id").head(limit).drop("_draw")
    kept_eval = eval_rows.join(keep, on="subject_id", how="semi")
    other = anchors.filter(pl.col("split") != eval_split)
    return pl.concat([other, kept_eval]).sort("subject_id")


@dataclass(frozen=True)
class ModelSpec:
    """One column of the results table."""

    name: str
    kind: str  # "lr" | "gbm" | "probe"
    checkpoint: Path | None = None
    random_init: bool = False

    @property
    def features(self) -> str:
        return "counts" if self.kind in ("lr", "gbm") else "cls+mean"


def parse_models(specs: Sequence[str]) -> list[ModelSpec]:
    """``lr``, ``gbm``, ``random_init``, ``ckpt:<path>`` -> :class:`ModelSpec`."""
    out: list[ModelSpec] = []
    checkpoints = [s.split(":", 1)[1] for s in specs if s.startswith("ckpt:")]
    for spec in specs:
        if spec == "lr":
            out.append(ModelSpec("lr", "lr"))
        elif spec == "gbm":
            out.append(ModelSpec("gbm", "gbm"))
        elif spec == "random_init":
            if not checkpoints:
                raise ValueError("random_init needs a ckpt: model to copy its architecture from")
            out.append(ModelSpec("random_init", "probe", Path(checkpoints[0]), random_init=True))
        elif spec.startswith("ckpt:"):
            path = Path(spec.split(":", 1)[1])
            out.append(ModelSpec(f"ckpt:{path.parent.name}", "probe", path))
        else:
            raise ValueError(f"unknown model spec {spec!r}")
    return out


def _split_arrays(anchors: pl.DataFrame, eval_split: str) -> dict[str, np.ndarray]:
    """Row indices of each split within the task frame."""
    split = anchors["split"].to_numpy()
    return {name: np.flatnonzero(split == name) for name in ("train", "tuning", eval_split)}


def _labels(anchors: pl.DataFrame) -> np.ndarray:
    return anchors["label"].to_numpy().astype(np.int64)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - not a git checkout
        return ""


def _count_matrix(
    reader: HistoryReader, anchors: pl.DataFrame, cache: Path | None
) -> sparse.csr_matrix:
    if cache is not None and cache.exists():
        stored = sparse.load_npz(cache)
        if stored.shape[0] == anchors.height:
            return stored.tocsr()
    matrix = baselines.count_matrix(reader, anchors)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cache, matrix)
    return matrix


def evaluate_task(
    task_name: str,
    anchors: pl.DataFrame,
    models: Sequence[ModelSpec],
    *,
    cache_dir: Path,
    source: str,
    feature_cache: Path | None,
    eval_split: str,
    n_boot: int,
    seed: int,
    device: str | None,
    few_shot: bool,
    predictions: list[pl.DataFrame] | None = None,
) -> dict:
    """Fit and score every model on one task's shared anchor frame.

    ``predictions`` collects one frame per (task, model) of the held-out scores,
    so a metric can be changed later without refitting anything.
    """
    predictions = [] if predictions is None else predictions
    index = _split_arrays(anchors, eval_split)
    y = _labels(anchors)
    entry: dict = {
        "counts": {name: int(rows.size) for name, rows in index.items()},
        "prevalence": {
            name: float(y[rows].mean()) if rows.size else float("nan")
            for name, rows in index.items()
        },
        "models": {},
        "paired": [],
    }
    y_tr, y_tu, y_ev = (y[index["train"]], y[index["tuning"]], y[index[eval_split]])
    if y_tr.min() == y_tr.max() or y_ev.min() == y_ev.max():
        entry["note"] = "single-class train or evaluation split; models not fit"
        return entry

    counts: sparse.csr_matrix | None = None
    keep: np.ndarray | None = None
    scores: dict[str, np.ndarray] = {}

    reader_counts = HistoryReader(cache_dir, max_len=None)
    for spec in models:
        started = time.time()
        if spec.kind in ("lr", "gbm"):
            if counts is None:
                cache = (
                    baselines.cache_path(feature_cache, source, task_name)
                    if feature_cache
                    else None
                )
                counts = _count_matrix(reader_counts, anchors, cache)
                keep = baselines.prune_columns(counts[index["train"]])
                log.info("%s: %d count columns kept of %d", task_name, keep.size, counts.shape[1])
            assert keep is not None
            matrix = counts[:, keep]
            x_tr, x_tu, x_ev = (
                matrix[index["train"]],
                matrix[index["tuning"]],
                matrix[index[eval_split]],
            )
            if spec.kind == "lr":
                fit = baselines.fit_logistic(x_tr, y_tr, x_tu, y_tu, seed=seed, name=spec.name)
            else:
                # Fit out of process; the result only holds predictions for x_ev.
                fit = baselines.fit_gbm(
                    x_tr, y_tr, x_tu, y_tu, x_predict=x_ev, seed=seed, name=spec.name
                )
            dense_for_few_shot = None
        else:
            cache = (
                probe.embedding_path(feature_cache, source, spec.name) if feature_cache else None
            )
            matrix = probe.embed_cached(
                spec.checkpoint,
                cache_dir,
                anchors,
                cache,
                random_init=spec.random_init,
                device=device,
                seed=seed,
            )
            x_tr, x_tu, x_ev = (
                matrix[index["train"]],
                matrix[index["tuning"]],
                matrix[index[eval_split]],
            )
            fit = probe.fit_probe(x_tr, y_tr, x_tu, y_tu, seed=seed, name=spec.name)
            dense_for_few_shot = (x_tr, x_tu, x_ev)

        p = fit.predict_proba(x_ev)
        scores[spec.name] = p
        predictions.append(
            anchors[index[eval_split]]
            .select("subject_id", "anchor_time", "label")
            .with_columns(
                task=pl.lit(task_name),
                model=pl.lit(spec.name),
                # float64 unconditionally: sklearn hands back the dtype of the
                # feature matrix, so the dense probes would otherwise give
                # float32 and refuse to stack with the sparse baselines.
                score=pl.Series(np.asarray(p, dtype=np.float64)),
            )
        )
        record = {
            "kind": spec.kind,
            "params": fit.params,
            "grid": fit.grid,
            "tuning_auroc": fit.tuning_auroc,
            "n_features": int(x_tr.shape[1]),
            "fit_seconds": round(time.time() - started, 2),
            "metrics": bootstrap_ci(y_ev, p, n_boot=n_boot, seed=seed),
        }
        if few_shot and spec.kind in ("lr", "probe"):
            if dense_for_few_shot is None:
                fs_tr, fs_tu, fs_ev = x_tr, x_tu, x_ev
                fit_fn = baselines.fit_logistic
            else:
                fs_tr, fs_tu, fs_ev = dense_for_few_shot
                fit_fn = probe.fit_probe
            record["few_shot"] = probe.few_shot(
                fs_tr, y_tr, fs_tu, y_tu, fs_ev, y_ev, fit_fn=fit_fn
            )
        entry["models"][spec.name] = record
        log.info(
            "%s / %s: auroc %.4f (%.1fs)",
            task_name,
            spec.name,
            record["metrics"]["auroc"]["point"],
            record["fit_seconds"],
        )

    names = list(scores)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            entry["paired"].append(
                {
                    "a": a,
                    "b": b,
                    "metric": "auroc",
                    **paired_bootstrap(
                        y_ev, scores[a], scores[b], metric="auroc", n_boot=n_boot, seed=seed
                    ),
                }
            )
    return entry


def run(
    source: str,
    models: Sequence[ModelSpec],
    out_dir: Path,
    *,
    meds_root: Path = Path("data/meds"),
    cache_root: Path = Path("data/cache"),
    task_root: Path = Path("data/tasks"),
    feature_cache: Path | None = Path("data/eval_cache"),
    task_names: Sequence[str] | None = None,
    eval_split: str = "held_out",
    n_boot: int = 1000,
    seed: int = 0,
    device: str | None = None,
    few_shot: bool = True,
    limit: int | None = None,
    eval_subject_limit: int | None = None,
    eval_subject_seed: int = 0,
) -> dict:
    """Build any missing task frames, then evaluate every model on every task."""
    started = time.time()
    meds_dir, cache_dir = meds_root / source, cache_root / source
    task_dir = task_root / source
    if not (task_dir / "tasks.json").exists():
        log.info("building tasks for %s", source)
        tasks.build_all(meds_dir, task_dir, names=task_names)
    summary = json.loads((task_dir / "tasks.json").read_text())

    supported, skipped = tasks.task_specs_for(meds_dir, task_names)
    results: dict = {
        "source": source,
        "meds_dir": str(meds_dir),
        "cache_dir": str(cache_dir),
        "task_dir": str(task_dir),
        "eval_split": eval_split,
        "anchor_seed": tasks.ANCHOR_SEED,
        "eval_subject_limit": eval_subject_limit,
        "eval_subject_seed": eval_subject_seed,
        "n_boot": n_boot,
        "seed": seed,
        "commit": _git_commit(),
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "models": {
            spec.name: {
                "kind": spec.kind,
                "checkpoint": str(spec.checkpoint) if spec.checkpoint else "",
                "features": spec.features,
                "random_init": spec.random_init,
            }
            for spec in models
        },
        "skipped": dict(skipped),
        "tasks": {},
    }
    reader = HistoryReader(cache_dir, max_len=None)
    predictions: list[pl.DataFrame] = []
    for spec in supported:
        if spec.name not in summary["tasks"]:
            results["skipped"][spec.name] = "not built"
            continue
        anchors = tasks.load_task(task_dir, spec.name)
        before = anchors.height
        anchors = reader.filter_present(anchors).sort("subject_id")
        if anchors.height != before:
            dropped = before - anchors.height
            log.info("%s: dropped %d anchors absent from the cache", spec.name, dropped)
        if limit:
            anchors = anchors.head(limit)
        anchors = restrict_eval_split(anchors, eval_split, eval_subject_limit, eval_subject_seed)
        results["tasks"][spec.name] = evaluate_task(
            spec.name,
            anchors,
            models,
            cache_dir=cache_dir,
            source=source,
            feature_cache=feature_cache,
            eval_split=eval_split,
            n_boot=n_boot,
            seed=seed,
            device=device,
            few_shot=few_shot,
            predictions=predictions,
        )
        results["tasks"][spec.name]["dropped_not_in_cache"] = before - anchors.height
    results["runtime_seconds"] = round(time.time() - started, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    if predictions:
        pl.concat(predictions).write_parquet(out_dir / "predictions.parquet")
    report.write(results, out_dir)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--models", default="lr,gbm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--meds-root", default="data/meds")
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument("--task-root", default="data/tasks")
    parser.add_argument("--feature-cache", default="data/eval_cache")
    parser.add_argument("--eval-split", default="held_out")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="first N anchors, for smoke tests")
    parser.add_argument(
        "--eval-subject-limit",
        type=int,
        default=None,
        help="restrict --eval-split to a seeded random subset of at most N subjects; "
        "train/tuning are untouched",
    )
    parser.add_argument("--eval-subject-seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("aces").setLevel(logging.WARNING)
    names = None if args.tasks == "all" else args.tasks.split(",")
    results = run(
        args.source,
        parse_models(args.models.split(",")),
        Path(args.out),
        meds_root=Path(args.meds_root),
        cache_root=Path(args.cache_root),
        task_root=Path(args.task_root),
        feature_cache=Path(args.feature_cache) if args.feature_cache else None,
        task_names=names,
        eval_split=args.eval_split,
        n_boot=args.bootstrap,
        seed=args.seed,
        device=args.device,
        few_shot=not args.no_few_shot,
        limit=args.limit,
        eval_subject_limit=args.eval_subject_limit,
        eval_subject_seed=args.eval_subject_seed,
    )
    print(f"wrote {Path(args.out) / 'results.md'} in {results['runtime_seconds']}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
