"""Summary statistics and sanity checks for a canonical MEDS directory.

``python -m ehrjepa.data.stats <meds_dir>`` prints the numbers we quote when
comparing sources: subject and event counts, vocabulary size, the events-per-
subject and per-subject-time-span distributions, how much of the stream carries a
numeric value, the top codes and the per-split breakdown. It also runs a handful
of integrity checks (events before birth, events after death, null times) because
every real EHR export has some, and it is better to know the count than to
discover it during training.

Everything is computed with lazy scans plus streaming group-bys, so it runs on
datasets much larger than memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import meds
import polars as pl

from ehrjepa.data.canonical import SPLITS

__all__ = ["compute_stats", "format_stats", "main", "scan_meds"]

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def scan_meds(meds_dir: str | Path, with_split: bool = True) -> pl.LazyFrame:
    """Lazily scan every shard under ``<meds_dir>/data``, tagged with its split."""
    root = Path(meds_dir)
    frames = []
    for split in SPLITS:
        shards = sorted((root / meds.data_subdirectory / split).glob("*.parquet"))
        if not shards:
            continue
        frame = pl.scan_parquet(shards)
        frames.append(frame.with_columns(pl.lit(split).alias("split")) if with_split else frame)
    if not frames:
        raise ValueError(f"no MEDS shards found under {root / meds.data_subdirectory}")
    return pl.concat(frames, how="vertical_relaxed")


def _quantiles(frame: pl.LazyFrame, column: str) -> dict[str, float | None]:
    result = frame.select(
        pl.col(column).quantile(0.1).alias("p10"),
        pl.col(column).median().alias("median"),
        pl.col(column).quantile(0.9).alias("p90"),
        pl.col(column).mean().alias("mean"),
    ).collect()
    return {k: (None if v is None else float(v)) for k, v in result.row(0, named=True).items()}


def compute_stats(meds_dir: str | Path, top_k: int = 20) -> dict[str, object]:
    """Compute the summary block for a canonical MEDS directory."""
    root = Path(meds_dir)
    data = scan_meds(root)

    totals = (
        data.select(
            pl.len().alias("events"),
            pl.col("subject_id").n_unique().alias("subjects"),
            pl.col("code").n_unique().alias("vocab_size"),
            pl.col("numeric_value").is_not_null().mean().alias("numeric_fraction"),
            pl.col("text_value").is_not_null().mean().alias("text_fraction"),
            pl.col("time").is_null().mean().alias("null_time_fraction"),
            pl.col("time").min().alias("time_min"),
            pl.col("time").max().alias("time_max"),
        )
        .collect()
        .row(0, named=True)
    )

    per_subject = data.group_by("subject_id").agg(
        pl.len().alias("n_events"),
        pl.col("time").min().alias("first"),
        pl.col("time").max().alias("last"),
    )
    span = per_subject.with_columns(
        ((pl.col("last") - pl.col("first")).dt.total_seconds() / _SECONDS_PER_YEAR).alias(
            "span_years"
        )
    )

    top_codes = (
        data.group_by("code")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_k)
        .collect()
    )

    by_split = (
        data.group_by("split")
        .agg(pl.len().alias("events"), pl.col("subject_id").n_unique().alias("subjects"))
        .collect()
        .sort("split")
    )

    splits_path = root / meds.subject_splits_filepath
    split_subjects = {}
    if splits_path.exists():
        split_subjects = dict(pl.read_parquet(splits_path).group_by("split").len().iter_rows())

    metadata_path = root / meds.dataset_metadata_filepath
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    return {
        "path": str(root),
        "dataset": metadata,
        "events": int(totals["events"]),
        "subjects": int(totals["subjects"]),
        "vocab_size": int(totals["vocab_size"]),
        "numeric_fraction": float(totals["numeric_fraction"] or 0.0),
        "text_fraction": float(totals["text_fraction"] or 0.0),
        "null_time_fraction": float(totals["null_time_fraction"] or 0.0),
        "time_min": str(totals["time_min"]),
        "time_max": str(totals["time_max"]),
        "events_per_subject": _quantiles(per_subject, "n_events"),
        "span_years_per_subject": _quantiles(span, "span_years"),
        "top_codes": top_codes.to_dicts(),
        "per_split": {
            row["split"]: {
                "events": row["events"],
                "subjects_with_events": row["subjects"],
                "subjects": split_subjects.get(row["split"]),
            }
            for row in by_split.to_dicts()
        },
        "checks": integrity_checks(root),
    }


def integrity_checks(meds_dir: str | Path) -> dict[str, int]:
    """Count the anomalies that are common enough to be worth quantifying."""
    data = scan_meds(meds_dir, with_split=False)
    anchors = (
        data.filter(pl.col("code").is_in([meds.birth_code, meds.death_code]))
        .group_by("subject_id", "code")
        .agg(pl.col("time").min().alias("t"))
        .collect()
        .pivot(on="code", index="subject_id", values="t")
    )
    for column in (meds.birth_code, meds.death_code):
        if column not in anchors.columns:
            anchors = anchors.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias(column))

    joined = data.join(anchors.lazy(), on="subject_id", how="left")
    counts = (
        joined.select(
            pl.col("time").is_null().sum().alias("null_time_events"),
            (pl.col("time") < pl.col(meds.birth_code)).sum().alias("events_before_birth"),
            (pl.col("time") > pl.col(meds.death_code)).sum().alias("events_after_death"),
            ((pl.col(meds.death_code) < pl.col(meds.birth_code)).sum()).alias(
                "events_with_death_before_birth"
            ),
        )
        .collect()
        .row(0, named=True)
    )

    subjects = data.select(pl.col("subject_id").n_unique()).collect().item()
    with_birth = anchors.filter(pl.col(meds.birth_code).is_not_null()).height
    return {
        **{k: int(v or 0) for k, v in counts.items()},
        "subjects_without_birth": int(subjects - with_birth),
        "subjects_with_death": int(anchors.filter(pl.col(meds.death_code).is_not_null()).height),
    }


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.3f}" if abs(value) < 1000 else f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def format_stats(stats: dict[str, object]) -> str:
    """Render :func:`compute_stats` output as markdown tables."""
    eps = stats["events_per_subject"]
    span = stats["span_years_per_subject"]
    summary = _table(
        ["metric", "value"],
        [
            ["dataset", stats["dataset"].get("dataset_name", "?")],
            ["path", stats["path"]],
            ["subjects", stats["subjects"]],
            ["events", stats["events"]],
            ["vocab size", stats["vocab_size"]],
            ["events/subject median", eps["median"]],
            ["events/subject p10", eps["p10"]],
            ["events/subject p90", eps["p90"]],
            ["span years/subject median", span["median"]],
            ["span years/subject p10", span["p10"]],
            ["span years/subject p90", span["p90"]],
            ["fraction with numeric_value", stats["numeric_fraction"]],
            ["fraction with text_value", stats["text_fraction"]],
            ["fraction with null time", stats["null_time_fraction"]],
            ["time range", f"{stats['time_min']} .. {stats['time_max']}"],
        ],
    )
    splits = _table(
        ["split", "subjects", "subjects with events", "events"],
        [
            [name, block["subjects"], block["subjects_with_events"], block["events"]]
            for name, block in stats["per_split"].items()
        ],
    )
    codes = _table(
        ["rank", "code", "count"],
        [[i + 1, row["code"], row["count"]] for i, row in enumerate(stats["top_codes"])],
    )
    checks = _table(["check", "count"], [[k, v] for k, v in stats["checks"].items()])
    return "\n\n".join(
        [
            "### Summary",
            summary,
            "### Per split",
            splits,
            f"### Top {len(stats['top_codes'])} codes",
            codes,
            "### Integrity checks",
            checks,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ehrjepa.data.stats",
        description="Summarise a canonical MEDS directory.",
    )
    parser.add_argument("meds_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of markdown")
    args = parser.parse_args(argv)

    stats = compute_stats(args.meds_dir, top_k=args.top_k)
    if args.json:
        json.dump(stats, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(format_stats(stats))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
