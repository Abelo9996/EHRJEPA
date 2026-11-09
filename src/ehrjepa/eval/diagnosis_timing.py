"""Audit whether new-diagnosis labels coincide with the current stay's discharge.

Uses existing task anchors/labels, first matching diagnosis times from canonical
MEDS, and hospital admission intervals from MIMIC. No model or GPU is involved.
Only aggregate counts are written. A timestamp match is evidence about label
timing, not proof of disease onset or leakage into model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from ehrjepa.eval.tasks import (
    CONFIG_DIR,
    TASKS,
    concept_matcher,
    dataset_family,
    load_concepts,
    load_task,
    task_path,
)

DIAGNOSIS_TASKS = tuple(t for t in TASKS if t.name.startswith("new_dx_365d/"))


def first_diagnoses(meds_dir: Path) -> pl.DataFrame:
    """Read only diagnosis rows; aggregate before materializing full-MIMIC data."""
    if dataset_family(meds_dir) != "mimic":
        raise ValueError("this audit requires MIMIC MEDS data")
    files = sorted(meds_dir.glob("data/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no MEDS shards in {meds_dir}")
    concepts = load_concepts()
    matchers = {
        t.name: concept_matcher(t.predicates["condition"], "mimic", concepts)
        for t in DIAGNOSIS_TASKS
    }
    first = (
        pl.scan_parquet(files)
        .select("subject_id", "time", "code")
        .filter(pl.any_horizontal(list(matchers.values())))
        .group_by("subject_id")
        .agg(
            [pl.col("time").filter(expr).min().alias(name) for name, expr in matchers.items()]
            + [pl.col("time").null_count().alias("_null_times")]
        )
        .collect(engine="streaming")
    )
    if first["_null_times"].sum():
        raise ValueError("matching MEDS diagnoses have null timestamps")
    return first.drop("_null_times")


def read_admissions(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        columns=["subject_id", "hadm_id", "admittime", "dischtime"],
        schema_overrides={"subject_id": pl.Int64, "hadm_id": pl.Int64},
        try_parse_dates=True,
    ).with_columns(pl.col("admittime", "dischtime").cast(pl.Datetime("us")))
    if frame.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("admissions contain missing IDs/times; audit cannot classify intervals")
    if frame["hadm_id"].n_unique() != frame.height:
        raise ValueError("admissions contain duplicate hadm_id values")
    if frame.filter(pl.col("dischtime") < pl.col("admittime")).height:
        raise ValueError("admissions contain discharge before admission")
    return frame


def classify_timing(
    labels: pl.DataFrame,
    diagnoses: pl.DataFrame,
    admissions: pl.DataFrame,
    *,
    horizon_days: int = 365,
) -> pl.DataFrame:
    """Classify rows without changing anchors, labels, or task definitions.

    ``diagnoses`` has one row per subject and a nullable ``first_diagnosis``.
    Stay membership is [admit, discharge); target is (anchor, anchor + horizon].
    Overlapping stays are counted once per label and flagged as ambiguous.
    """
    required = labels.select("subject_id", "anchor_time", "label", "split")
    if required.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("labels contain null values")
    if labels.select("subject_id", "anchor_time").unique().height != labels.height:
        raise ValueError("duplicate subject/anchor labels")
    if not labels["label"].is_in([0, 1]).all():
        raise ValueError("labels must be binary")
    if diagnoses["subject_id"].n_unique() != diagnoses.height:
        raise ValueError("first diagnosis must be unique per subject")
    rows = required.with_row_index("_row").join(diagnoses, on="subject_id", how="left")
    rows = rows.with_columns(
        subject_has_admission=pl.col("subject_id").is_in(
            admissions["subject_id"].unique().implode()
        ),
        prior_diagnosis=(pl.col("first_diagnosis") <= pl.col("anchor_time")).fill_null(False),
        diagnosis_in_window=(
            (pl.col("first_diagnosis") > pl.col("anchor_time"))
            & (pl.col("first_diagnosis") <= pl.col("anchor_time") + pl.duration(days=horizon_days))
        ).fill_null(False),
    ).with_columns(
        label_mismatch=pl.col("prior_diagnosis")
        | (pl.col("label") != pl.col("diagnosis_in_window").cast(pl.Int8))
    )
    active = (
        rows.join(admissions, on="subject_id", how="inner")
        .filter(
            (pl.col("admittime") <= pl.col("anchor_time"))
            & (pl.col("anchor_time") < pl.col("dischtime"))
        )
        .group_by("_row")
        .agg(
            active_stays=pl.col("hadm_id").n_unique(),
            strictly_in_stay=(pl.col("admittime") < pl.col("anchor_time")).any(),
            first_at_current_discharge=(pl.col("first_diagnosis") == pl.col("dischtime")).any(),
        )
    )
    return rows.join(active, on="_row", how="left").with_columns(
        pl.col("active_stays").fill_null(0),
        pl.col("strictly_in_stay", "first_at_current_discharge").fill_null(False),
    )


def summarize(rows: pl.DataFrame) -> dict:
    positives = rows.filter(pl.col("label") == 1)
    matching = positives.filter(
        pl.col("first_at_current_discharge") & pl.col("diagnosis_in_window")
    )
    return {
        "label_rows": rows.height,
        "positives": positives.height,
        "subjects_without_admission_rows": rows.filter(~pl.col("subject_has_admission")).height,
        "anchors_in_stay": rows.filter(pl.col("active_stays") > 0).height,
        "anchors_strictly_in_stay": int(rows["strictly_in_stay"].sum()),
        "anchors_with_overlapping_stays": rows.filter(pl.col("active_stays") > 1).height,
        "positives_at_current_stay_discharge": matching.height,
        "fraction_of_positives_at_current_stay_discharge": (
            matching.height / positives.height if positives.height else None
        ),
        "label_mismatches": int(rows["label_mismatch"].sum()),
        "rows_with_diagnosis_at_or_before_anchor": int(rows["prior_diagnosis"].sum()),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(meds_dir: Path, tasks_dir: Path, admissions_path: Path) -> dict:
    diagnoses = first_diagnoses(meds_dir)
    admissions = read_admissions(admissions_path)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "meds_dir": str(meds_dir),
        "tasks_dir": str(tasks_dir),
        "admissions_path": str(admissions_path),
        "admissions_sha256": _sha256(admissions_path),
        "audit_code_sha256": _sha256(Path(__file__)),
        "predicates_sha256": _sha256(CONFIG_DIR / "predicates.yaml"),
        "task_config_sha256": _sha256(CONFIG_DIR / "new_dx_365d.yaml"),
        "interpretation": (
            "First matching MEDS diagnosis coincides with discharge of a hospital stay "
            "containing the anchor. This is a timing audit, not clinical onset adjudication "
            "or a feature-leakage test. Pooled counts are task-label rows, not unique patients."
        ),
        "tasks": {},
    }
    pooled = []
    for spec in DIAGNOSIS_TASKS:
        labels = load_task(tasks_dir, spec.name)
        first = diagnoses.select("subject_id", first_diagnosis=pl.col(spec.name))
        rows = classify_timing(labels, first, admissions, horizon_days=spec.horizon_days)
        result["tasks"][spec.name] = {
            "labels_sha256": _sha256(task_path(tasks_dir, spec.name)),
            "all": summarize(rows),
            "by_split": {
                split: summarize(rows.filter(pl.col("split") == split))
                for split in sorted(rows["split"].unique())
            },
        }
        pooled.append(rows)
    result["pooled_task_rows"] = summarize(pl.concat(pooled))
    result["labels_consistent"] = result["pooled_task_rows"]["label_mismatches"] == 0
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meds-dir", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--admissions", type=Path, required=True, help="hosp/admissions.csv[.gz]")
    parser.add_argument("--out", type=Path, required=True, help="aggregate JSON; no patient rows")
    args = parser.parse_args(argv)
    report = audit(args.meds_dir, args.tasks_dir, args.admissions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, task in report["tasks"].items():
        counts = task["all"]
        print(
            f"{name}: {counts['positives_at_current_stay_discharge']}/"
            f"{counts['positives']} positives at current-stay discharge; "
            f"{counts['label_mismatches']} inconsistent labels"
        )
    return 0 if report["labels_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
