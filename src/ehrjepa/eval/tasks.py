"""Downstream task definitions: anchors and labels, from MEDS, through ACES.

Labels are computed from the **MEDS extract**, never from the tensor cache: the
cache rolls rare codes up to an ancestor, so ``ICD9CM//25001`` and
``ICD9CM//25002`` can share an id there. A label built on rolled-up ids would
not be the label its name claims.

The windowing and the label predicate are evaluated by `ACES
<https://github.com/justin13601/ACES>`_ (Xu et al., *Automatic Cohort
Extraction System for Event-Stream Datasets*), through its ``direct`` data
standard: this module builds a predicates dataframe -- one row per
``(subject_id, timestamp)``, one integer column per concept -- and hands it to
:func:`aces.query.query` together with a task YAML from ``configs/tasks/``. What
ACES cannot express, and what therefore lives here, is the **anchor rule**: its
triggers are predicate matches on the data, so "one seeded random event time per
subject, with at least 32 events before it and enough record after it" has to be
materialised as a synthetic ``anchor`` predicate before ACES runs.

Anchor rule (default)
    One anchor per subject: a uniformly random event time ``t`` among the
    subject's distinct event times such that

    * at least ``min_history`` events fall strictly before ``t``;
    * ``t`` is strictly before ``MEDS_DEATH`` when the subject has one;
    * the record extends at least ``horizon`` past ``t``, **or** the subject
      dies within ``horizon`` of ``t`` (otherwise the label would be censored
      rather than negative).

    The draw is a pure function of ``(seed, subject_id)`` via ``blake2b``, the
    same construction the split assignment uses, so it does not depend on row
    order, shard count, or polars version.

    ``readmission_30d`` overrides the candidate set: anchors are drawn from
    inpatient discharge events only. The censoring horizon always matches the
    task's own label window, so a 30-day task requires 30 days of follow-up.

Because anchors precede death, and history is everything strictly before the
anchor, no event at or after ``MEDS_DEATH`` can ever enter a history window.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import yaml

__all__ = [
    "TaskSpec",
    "TASKS",
    "build_task",
    "dataset_family",
    "read_events",
    "select_anchors",
    "task_specs_for",
]

log = logging.getLogger(__name__)

#: Repository-relative location of the ACES task YAMLs and the concept table.
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "tasks"

#: Minimum events strictly before an anchor.
MIN_HISTORY = 32

#: Seed for the anchor draw. Changing it reshuffles every task ever produced.
ANCHOR_SEED = 20260903

_CHRONIC = {
    "diabetes": "dx_diabetes",
    "heart_failure": "dx_heart_failure",
    "ckd": "dx_ckd",
    "copd": "dx_copd",
}


@dataclass(frozen=True)
class TaskSpec:
    """One task: which ACES YAML, which concepts, and how the anchor is drawn."""

    name: str
    config: str
    horizon_days: int
    #: ACES predicate name -> concept name in ``configs/tasks/predicates.yaml``.
    predicates: Mapping[str, str]
    #: Concept whose events are the only allowed anchors, or ``None`` for any event.
    anchor_concept: str | None = None
    #: Concepts that must exist in the source for this task to be defined.
    requires: tuple[str, ...] = field(default=())

    @property
    def needed_concepts(self) -> tuple[str, ...]:
        names = {c for p, c in self.predicates.items() if p != "anchor"}
        names.update(self.requires)
        if self.anchor_concept:
            names.add(self.anchor_concept)
        return tuple(sorted(names))


def _default_specs() -> tuple[TaskSpec, ...]:
    specs = [
        TaskSpec(
            name="mortality_365d",
            config="mortality_365d.yaml",
            horizon_days=365,
            predicates={"death": "death"},
            requires=("death",),
        ),
        TaskSpec(
            name="inpatient_365d",
            config="inpatient_365d.yaml",
            horizon_days=365,
            predicates={"inpatient_admission": "inpatient_admission"},
            requires=("inpatient_admission",),
        ),
        TaskSpec(
            name="readmission_30d",
            config="readmission_30d.yaml",
            horizon_days=30,
            predicates={"inpatient_admission": "inpatient_admission"},
            anchor_concept="inpatient_discharge",
            requires=("inpatient_admission", "inpatient_discharge"),
        ),
    ]
    specs += [
        TaskSpec(
            name=f"new_dx_365d/{short}",
            config="new_dx_365d.yaml",
            horizon_days=365,
            predicates={"condition": concept},
            requires=(concept,),
        )
        for short, concept in _CHRONIC.items()
    ]
    return tuple(specs)


#: Every task this module knows how to build, in report order.
TASKS: tuple[TaskSpec, ...] = _default_specs()


# --------------------------------------------------------------------------- #
# Concept table and source family


def load_concepts(config_dir: Path | str = CONFIG_DIR) -> dict:
    """The parsed ``predicates.yaml`` concept table."""
    with open(Path(config_dir) / "predicates.yaml") as handle:
        return yaml.safe_load(handle)


def dataset_family(meds_dir: Path | str, config_dir: Path | str = CONFIG_DIR) -> str:
    """``desynpuf`` / ``synthea`` / ``mimic`` for a MEDS extract."""
    with open(Path(meds_dir) / "metadata" / "dataset.json") as handle:
        name = json.load(handle)["dataset_name"]
    families = load_concepts(config_dir)["dataset_families"]
    if name not in families:
        raise KeyError(f"no source family for dataset_name {name!r}; add it to predicates.yaml")
    return families[name]


def concept_matcher(concept: str, family: str, concepts: Mapping) -> pl.Expr | None:
    """A boolean expression over ``code``, or ``None`` if the source lacks the concept."""
    entry = concepts["concepts"].get(concept, {}).get(family)
    if not entry:
        return None
    terms: list[pl.Expr] = []
    if entry.get("codes"):
        terms.append(pl.col("code").is_in(list(entry["codes"])))
    for prefix in entry.get("prefixes", ()):
        terms.append(pl.col("code").str.starts_with(prefix))
    if not terms:
        return None
    expr = terms[0]
    for term in terms[1:]:
        expr = expr | term
    return expr


def task_specs_for(
    meds_dir: Path | str, names: Sequence[str] | None = None, config_dir: Path | str = CONFIG_DIR
) -> tuple[list[TaskSpec], dict[str, str]]:
    """Split :data:`TASKS` into the ones this source supports and the ones it does not.

    Returns ``(supported, skipped)`` where ``skipped`` maps a task name to the
    concept the source has no codes for.
    """
    family = dataset_family(meds_dir, config_dir)
    concepts = load_concepts(config_dir)
    wanted = list(TASKS) if names is None else [t for t in TASKS if t.name in set(names)]
    if names is not None:
        unknown = set(names) - {t.name for t in TASKS}
        if unknown:
            raise ValueError(f"unknown tasks: {sorted(unknown)}")
    supported, skipped = [], {}
    for spec in wanted:
        missing = [c for c in spec.needed_concepts if concept_matcher(c, family, concepts) is None]
        if missing:
            skipped[spec.name] = missing[0]
        else:
            supported.append(spec)
    return supported, skipped


# --------------------------------------------------------------------------- #
# Events


def read_events(meds_dir: Path | str) -> pl.DataFrame:
    """Every event of every split, with null times resolved the way the cache does.

    ``event_time`` is ``coalesce(time, MEDS_BIRTH time, first event time)`` --
    exactly the rule :mod:`ehrjepa.data.cache` uses for ``time_min`` -- so an
    anchor chosen here indexes the same position in the tensor cache.
    """
    meds_dir = Path(meds_dir)
    frames = [pl.scan_parquet(p) for p in sorted(meds_dir.glob("data/*/*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"no MEDS shards under {meds_dir / 'data'}")
    events = pl.concat(frames).select("subject_id", "time", "code")
    anchors = events.group_by("subject_id").agg(
        birth=pl.col("time").filter(pl.col("code") == "MEDS_BIRTH").min(),
        first=pl.col("time").min(),
    )
    anchors = anchors.select("subject_id", anchor=pl.coalesce("birth", "first"))
    return (
        events.join(anchors, on="subject_id", how="left")
        .with_columns(event_time=pl.coalesce("time", "anchor"))
        .select("subject_id", "event_time", "code")
        .collect()
    )


def _hash_u64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


# --------------------------------------------------------------------------- #
# Anchors


def select_anchors(
    events: pl.DataFrame,
    horizon_days: int,
    *,
    death_mask: pl.Expr | None = None,
    candidate_mask: pl.Expr | None = None,
    min_history: int = MIN_HISTORY,
    seed: int = ANCHOR_SEED,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """One anchor per subject, plus the counts of who was excluded and why.

    ``events`` needs ``subject_id``, ``event_time`` and ``code``. ``death_mask``
    and ``candidate_mask`` are expressions over ``code``.
    """
    horizon = pl.duration(days=horizon_days)
    n_subjects = events["subject_id"].n_unique()

    per_time = (
        events.group_by("subject_id", "event_time")
        .len()
        .sort("subject_id", "event_time")
        .with_columns(
            n_before=pl.col("len").cum_sum().over("subject_id") - pl.col("len"),
            last_time=pl.col("event_time").max().over("subject_id"),
        )
    )
    if death_mask is None:
        per_time = per_time.with_columns(death_time=pl.lit(None, dtype=pl.Datetime("us")))
    else:
        deaths = (
            events.filter(death_mask)
            .group_by("subject_id")
            .agg(death_time=pl.col("event_time").min())
        )
        per_time = per_time.join(deaths, on="subject_id", how="left")

    if candidate_mask is not None:
        allowed = events.filter(candidate_mask).select("subject_id", "event_time").unique()
        per_time = per_time.join(allowed, on=["subject_id", "event_time"], how="semi")
    n_with_candidates = per_time["subject_id"].n_unique()

    has_history = per_time.filter(pl.col("n_before") >= min_history)
    n_with_history = has_history["subject_id"].n_unique()

    before_death = has_history.filter(
        pl.col("death_time").is_null() | (pl.col("event_time") < pl.col("death_time"))
    )
    n_before_death = before_death["subject_id"].n_unique()

    eligible = before_death.filter(
        ((pl.col("last_time") - pl.col("event_time")) >= horizon)
        | (
            pl.col("death_time").is_not_null()
            & ((pl.col("death_time") - pl.col("event_time")) <= horizon)
        )
    )
    n_eligible = eligible["subject_id"].n_unique()

    subjects = eligible["subject_id"].unique().sort().to_list()
    draws = pl.DataFrame(
        {
            "subject_id": subjects,
            "_draw": [_hash_u64(f"anchor:{seed}:{s}") for s in subjects],
        },
        schema={"subject_id": events.schema["subject_id"], "_draw": pl.UInt64},
    )
    anchors = (
        eligible.sort("subject_id", "event_time")
        .with_columns(
            _rank=pl.int_range(pl.len()).over("subject_id"),
            _n=pl.len().over("subject_id"),
        )
        .join(draws, on="subject_id", how="inner")
        .filter(pl.col("_rank") == (pl.col("_draw") % pl.col("_n")))
        .select("subject_id", anchor_time=pl.col("event_time"))
    )
    counts = {
        "subjects": n_subjects,
        "with_candidate_events": n_with_candidates,
        "with_min_history": n_with_history,
        "with_anchor_before_death": n_before_death,
        "with_followup_or_death": n_eligible,
        "anchored": anchors.height,
    }
    return anchors, counts


# --------------------------------------------------------------------------- #
# ACES


def build_predicates_frame(
    events: pl.DataFrame,
    anchors: pl.DataFrame,
    predicates: Mapping[str, pl.Expr],
) -> pl.DataFrame:
    """One row per ``(subject_id, timestamp)``, one integer column per predicate.

    This is the ``direct`` predicates dataframe ACES consumes. ``anchor`` is a
    synthetic predicate set to 1 at exactly the chosen anchor times.
    """
    columns = [expr.cast(pl.Int32).alias(name) for name, expr in predicates.items()]
    frame = (
        events.with_columns(columns)
        .group_by("subject_id", "event_time")
        .agg(
            [pl.col(name).sum().cast(pl.Int32) for name in predicates],
            _ANY_EVENT=pl.len().cast(pl.Int32),
        )
        .rename({"event_time": "timestamp"})
    )
    marked = anchors.select("subject_id", timestamp=pl.col("anchor_time"), _anchor=pl.lit(1))
    return (
        frame.join(marked, on=["subject_id", "timestamp"], how="left")
        .with_columns(anchor=pl.col("_anchor").fill_null(0).cast(pl.Int32))
        .drop("_anchor")
        .sort("subject_id", "timestamp")
    )


def run_aces(config_path: Path | str, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """``(subject_id, anchor_time, label)`` from an ACES task YAML."""
    from aces.config import TaskExtractorConfig
    from aces.query import query

    cfg = TaskExtractorConfig.load(Path(config_path))
    result = query(cfg, predicates_df)
    if result.height == 0:
        return pl.DataFrame(
            schema={
                "subject_id": predicates_df.schema["subject_id"],
                "anchor_time": pl.Datetime("us"),
                "label": pl.Int8,
            }
        )
    return result.select(
        "subject_id",
        anchor_time=pl.col("index_timestamp"),
        label=(pl.col("label") > 0).cast(pl.Int8),
    )


# --------------------------------------------------------------------------- #
# Driver


def build_task(
    spec: TaskSpec,
    meds_dir: Path | str,
    *,
    events: pl.DataFrame | None = None,
    config_dir: Path | str = CONFIG_DIR,
    seed: int = ANCHOR_SEED,
    min_history: int = MIN_HISTORY,
) -> tuple[pl.DataFrame, dict]:
    """Build one task's ``(subject_id, anchor_time, label, split)`` frame."""
    meds_dir = Path(meds_dir)
    config_dir = Path(config_dir)
    family = dataset_family(meds_dir, config_dir)
    concepts = load_concepts(config_dir)
    if events is None:
        events = read_events(meds_dir)

    def matcher(concept: str) -> pl.Expr:
        expr = concept_matcher(concept, family, concepts)
        if expr is None:
            raise KeyError(f"source family {family!r} has no codes for concept {concept!r}")
        return expr

    death = concept_matcher("death", family, concepts)
    candidate = matcher(spec.anchor_concept) if spec.anchor_concept else None
    anchors, counts = select_anchors(
        events,
        spec.horizon_days,
        death_mask=death,
        candidate_mask=candidate,
        min_history=min_history,
        seed=seed,
    )
    predicates = {name: matcher(concept) for name, concept in spec.predicates.items()}
    frame = build_predicates_frame(events, anchors, predicates)
    labels = run_aces(config_dir / spec.config, frame)
    counts["labelled"] = labels.height

    splits = pl.read_parquet(meds_dir / "metadata" / "subject_splits.parquet")
    labelled = labels.join(splits, on="subject_id", how="inner").select(
        "subject_id", "anchor_time", "label", "split"
    )
    if labelled.height != labels.height:  # pragma: no cover - split table is exhaustive
        raise ValueError("some labelled subjects have no split assignment")
    if labelled["subject_id"].n_unique() != labelled.height:
        raise ValueError(f"task {spec.name} produced more than one anchor per subject")

    counts["prevalence"] = {
        row["split"]: {"n": row["n"], "positives": row["pos"], "rate": row["rate"]}
        for row in labelled.group_by("split")
        .agg(n=pl.len(), pos=pl.col("label").sum(), rate=pl.col("label").mean())
        .sort("split")
        .to_dicts()
    }
    counts["task"] = spec.name
    counts["source_family"] = family
    counts["seed"] = seed
    counts["horizon_days"] = spec.horizon_days
    return labelled.sort("subject_id"), counts


def build_all(
    meds_dir: Path | str,
    out_dir: Path | str,
    *,
    names: Sequence[str] | None = None,
    config_dir: Path | str = CONFIG_DIR,
    seed: int = ANCHOR_SEED,
) -> dict:
    """Build every supported task for a source and write one parquet per task."""
    meds_dir, out_dir = Path(meds_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    supported, skipped = task_specs_for(meds_dir, names, config_dir)
    events = read_events(meds_dir)
    summary: dict = {"source": meds_dir.name, "skipped": skipped, "tasks": {}}
    for spec in supported:
        labels, counts = build_task(
            spec, meds_dir, events=events, config_dir=config_dir, seed=seed
        )
        path = out_dir / f"{spec.name.replace('/', '__')}.parquet"
        labels.write_parquet(path)
        counts["path"] = str(path)
        summary["tasks"][spec.name] = counts
        log.info("%s: %d rows, prevalence %s", spec.name, labels.height, counts["prevalence"])
    with open(out_dir / "tasks.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def task_path(out_dir: Path | str, name: str) -> Path:
    """Where :func:`build_all` writes the parquet for a task name."""
    return Path(out_dir) / f"{name.replace('/', '__')}.parquet"


def load_task(out_dir: Path | str, name: str) -> pl.DataFrame:
    """Read a built task frame."""
    return pl.read_parquet(task_path(out_dir, name))


def _cli(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="name under data/meds/")
    parser.add_argument("--meds-root", default="data/meds")
    parser.add_argument("--out", default=None, help="default: data/tasks/<source>")
    parser.add_argument("--tasks", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--seed", type=int, default=ANCHOR_SEED)
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names = None if args.tasks == "all" else args.tasks.split(",")
    out = Path(args.out) if args.out else Path("data/tasks") / args.source
    summary = build_all(
        Path(args.meds_root) / args.source, out, names=names, seed=args.seed
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
