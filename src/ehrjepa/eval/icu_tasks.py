"""Hourly ICU tasks on the two PhysioNet challenge sources, in native polars.

The three longitudinal tasks in :mod:`ehrjepa.eval.tasks` are built through ACES
against a predicates dataframe, with one seeded anchor per subject drawn from
whatever event times the subject happens to have. Neither half of that fits a
single ICU stay. The anchor is not a random event time -- it is a fixed *hour of
the stay* (hour 12, hour 24, hour 48) or a grid of them -- and the horizon is
hours, not days, which is below the resolution of every ACES config here. So
these builders compute anchors and labels directly.

What is *not* different is the anchor discipline, which is the part that makes
the numbers mean anything:

* history is everything strictly before the anchor timestamp -- the cut is
  :meth:`~ehrjepa.data.dataset.EventSequenceDataset.windows_at`, the same one
  every other task uses, and it is exclusive at the anchor;
* the labelling event is always strictly after the anchor. For sepsis that is
  guaranteed by dropping anchors at or after onset; for mortality it is
  guaranteed by the ETL, which parks every outcome event an hour past the end of
  the 48-hour observation window (see
  :mod:`ehrjepa.data.etl.physionet2012`);
* anchors are a pure function of ``(seed, subject_id)`` where any choice is
  made, via the same ``blake2b`` construction as the split assignment.

Because time in both sources is synthetic and shared -- hour 0 of every stay is
``2100-01-01T00:00`` -- an "hour of the stay" is just an offset from that
origin, and the timestamps a task emits are ordinary ``Datetime("us")`` values
that ``windows_at`` bisects like any other.

Multiple anchors per stay
-------------------------
``sepsis_6h`` emits up to :data:`MAX_ANCHORS_PER_STAY` anchors per stay, so its
rows are not independent. Splits stay patient-disjoint (they are a function of
``subject_id``, and every anchor of a stay shares it), so there is no train/test
leakage, but a row-level bootstrap will understate the width of the interval.
That is a property of the task, recorded here and in the run's ``counts``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from ehrjepa.data.etl.physionet2012 import ORIGIN as ORIGIN_2012
from ehrjepa.data.etl.physionet2019 import ORIGIN as ORIGIN_2019

__all__ = [
    "BUILDERS",
    "MAX_ANCHORS_PER_STAY",
    "SEPSIS_LEAD_HOURS",
    "build_mortality",
    "build_sepsis_6h",
    "build_sepsis_stay",
    "hour_grid",
]

#: The onset window of ``sepsis_6h``: label 1 when onset falls in ``(h, h + 6]``.
SEPSIS_LEAD_HOURS = 6

#: Earliest anchor hour of ``sepsis_6h``. Six hours of stay must precede it.
SEPSIS_MIN_HOUR = 6

#: Anchor hour of ``sepsis_stay``.
SEPSIS_STAY_HOUR = 12

#: Cap on the anchors drawn per stay for ``sepsis_6h``. Without it a 300-hour
#: stay would contribute 300 correlated rows and a 12-hour stay one.
MAX_ANCHORS_PER_STAY = 8

#: Measurement events required strictly before a mortality anchor. A 2012 record
#: that never got that far is a record with nothing to predict from, not a
#: negative.
MORTALITY_MIN_EVENTS = 10


def _hash_u64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


def hour_grid(origin: dt.datetime, hours: pl.Expr) -> pl.Expr:
    """Turn an integer hour-of-stay expression into a synthetic-clock timestamp."""
    return (pl.lit(origin) + pl.duration(hours=hours)).cast(pl.Datetime("us"))


def _hours_since(origin: dt.datetime, time: pl.Expr) -> pl.Expr:
    """Whole hours between ``origin`` and ``time``, floored."""
    return (time - pl.lit(origin)).dt.total_minutes() // 60


def _splits(meds_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(meds_dir / "metadata" / "subject_splits.parquet")


def _finish(
    labels: pl.DataFrame, meds_dir: Path, counts: dict[str, object]
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Attach splits, order rows, and record the per-split prevalence."""
    labelled = (
        labels.join(_splits(meds_dir), on="subject_id", how="inner")
        .select("subject_id", "anchor_time", pl.col("label").cast(pl.Int8), "split")
        .sort("subject_id", "anchor_time")
    )
    if labelled.height != labels.height:  # pragma: no cover - split table is exhaustive
        raise ValueError("some labelled subjects have no split assignment")
    counts["labelled"] = labelled.height
    counts["anchored_subjects"] = labelled["subject_id"].n_unique()
    counts["prevalence"] = {
        row["split"]: {"n": row["n"], "positives": row["pos"], "rate": row["rate"]}
        for row in labelled.group_by("split")
        .agg(n=pl.len(), pos=pl.col("label").sum(), rate=pl.col("label").mean())
        .sort("split")
        .to_dicts()
    }
    return labelled, counts


# --------------------------------------------------------------------------- #
# PhysioNet 2019: sepsis
# --------------------------------------------------------------------------- #


def _stay_hours_and_onset(events: pl.DataFrame) -> pl.DataFrame:
    """Per stay: the last recorded hour and the sepsis onset hour, if any."""
    hour = _hours_since(ORIGIN_2019, pl.col("event_time"))
    return (
        events.filter(pl.col("event_time") >= pl.lit(ORIGIN_2019))
        .group_by("subject_id")
        .agg(
            last_hour=hour.max(),
            onset_hour=hour.filter(pl.col("code") == "SEPSIS_ONSET").min(),
        )
        .sort("subject_id")
    )


def build_sepsis_6h(
    events: pl.DataFrame, meds_dir: Path, *, seed: int
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Early sepsis prediction on an hourly grid.

    One candidate anchor at every stay hour ``h >= 6`` that has at least six
    hours of stay behind it and is strictly before onset; label 1 when onset
    falls in ``(h, h + 6]``. Candidates are thinned to at most
    :data:`MAX_ANCHORS_PER_STAY` per stay by a seeded hash of
    ``(seed, subject_id, hour)``, keeping the lowest hashes, so the kept set is a
    pure function of the seed and the stay and does not depend on row order.

    Anchors at or after onset are dropped rather than labelled 0: after onset the
    question the challenge asks -- how early can onset be called -- has no
    answer, and the post-onset hours of a septic stay would otherwise enter the
    negative class carrying every physiological sign of sepsis.
    """
    stays = _stay_hours_and_onset(events)
    counts: dict[str, object] = {
        "subjects": events["subject_id"].n_unique(),
        "septic_subjects": int(stays["onset_hour"].is_not_null().sum()),
        "max_anchors_per_stay": MAX_ANCHORS_PER_STAY,
        "lead_hours": SEPSIS_LEAD_HOURS,
        "multi_anchor": True,
    }

    candidates = (
        stays.filter(pl.col("last_hour") >= SEPSIS_MIN_HOUR)
        .with_columns(
            hour=pl.int_ranges(pl.lit(SEPSIS_MIN_HOUR, dtype=pl.Int64), pl.col("last_hour") + 1)
        )
        .explode("hour", empty_as_null=False)
        .filter(pl.col("onset_hour").is_null() | (pl.col("hour") < pl.col("onset_hour")))
    )
    counts["candidate_anchors"] = candidates.height
    counts["subjects_with_candidates"] = candidates["subject_id"].n_unique()

    draws = pl.Series(
        "_draw",
        [
            _hash_u64(f"icu_anchor:{seed}:{s}:{h}")
            for s, h in zip(
                candidates["subject_id"].to_list(), candidates["hour"].to_list(), strict=True
            )
        ],
        dtype=pl.UInt64,
    )
    kept = (
        candidates.with_columns(draws)
        .sort("subject_id", "_draw", "hour")
        .with_columns(_rank=pl.int_range(pl.len()).over("subject_id"))
        .filter(pl.col("_rank") < MAX_ANCHORS_PER_STAY)
    )
    labels = kept.select(
        "subject_id",
        anchor_time=hour_grid(ORIGIN_2019, pl.col("hour")),
        label=(
            pl.col("onset_hour").is_not_null()
            & (pl.col("onset_hour") <= pl.col("hour") + SEPSIS_LEAD_HOURS)
        ).cast(pl.Int8),
    )
    return _finish(labels, meds_dir, counts)


def build_sepsis_stay(
    events: pl.DataFrame, meds_dir: Path, *, seed: int
) -> tuple[pl.DataFrame, dict[str, object]]:
    """One anchor per stay at hour 12; label = sepsis anywhere in the stay.

    Stays whose onset is at or before hour 12 are dropped: their anchor would sit
    after the event it is meant to predict, and the label event itself would be
    inside the history window.
    """
    stays = _stay_hours_and_onset(events)
    counts: dict[str, object] = {
        "subjects": events["subject_id"].n_unique(),
        "septic_subjects": int(stays["onset_hour"].is_not_null().sum()),
        "anchor_hour": SEPSIS_STAY_HOUR,
        "multi_anchor": False,
    }
    eligible = stays.filter(pl.col("last_hour") >= SEPSIS_STAY_HOUR)
    counts["long_enough"] = eligible.height
    eligible = eligible.filter(
        pl.col("onset_hour").is_null() | (pl.col("onset_hour") > SEPSIS_STAY_HOUR)
    )
    counts["onset_after_anchor"] = eligible.height
    labels = eligible.select(
        "subject_id",
        anchor_time=hour_grid(ORIGIN_2019, pl.lit(SEPSIS_STAY_HOUR, dtype=pl.Int64)),
        label=pl.col("onset_hour").is_not_null().cast(pl.Int8),
    )
    return _finish(labels, meds_dir, counts)


# --------------------------------------------------------------------------- #
# PhysioNet 2012: in-hospital mortality
# --------------------------------------------------------------------------- #


def build_mortality(
    events: pl.DataFrame, meds_dir: Path, *, seed: int, anchor_hour: int
) -> tuple[pl.DataFrame, dict[str, object]]:
    """In-hospital mortality, called from a fixed hour of the 48-hour window.

    The label is ``MEDS_DEATH``, which the ETL places at hour 49 -- past every
    admissible anchor -- so history strictly before hour 24 or hour 48 can never
    contain it. A record needs at least :data:`MORTALITY_MIN_EVENTS` events
    strictly before the anchor to be admitted; the 2012 release nominally covers
    48 hours for every record, but a handful are near-empty, and a record with no
    observations is not a negative.
    """
    anchor_time = ORIGIN_2012 + dt.timedelta(hours=anchor_hour)
    counts: dict[str, object] = {
        "subjects": events["subject_id"].n_unique(),
        "anchor_hour": anchor_hour,
        "min_events_before_anchor": MORTALITY_MIN_EVENTS,
        "multi_anchor": False,
    }
    per_subject = events.group_by("subject_id").agg(
        n_before=(pl.col("event_time") < pl.lit(anchor_time)).sum(),
        died=(pl.col("code") == "MEDS_DEATH").any(),
    )
    counts["deaths"] = int(per_subject["died"].sum())
    eligible = per_subject.filter(pl.col("n_before") >= MORTALITY_MIN_EVENTS)
    counts["with_min_history"] = eligible.height
    labels = eligible.select(
        "subject_id",
        anchor_time=pl.lit(anchor_time).cast(pl.Datetime("us")),
        label=pl.col("died").cast(pl.Int8),
    )
    return _finish(labels, meds_dir, counts)


def _mortality_at(hour: int):
    def builder(events, meds_dir, *, seed):
        return build_mortality(events, meds_dir, seed=seed, anchor_hour=hour)

    builder.__doc__ = f"``build_mortality`` with ``anchor_hour={hour}``."
    return builder


#: Builder name -> callable, as :class:`ehrjepa.eval.tasks.TaskSpec.builder` names it.
BUILDERS: Mapping[str, object] = {
    "sepsis_6h": build_sepsis_6h,
    "sepsis_stay": build_sepsis_stay,
    "mortality_inhospital_24h": _mortality_at(24),
    "mortality_inhospital_48h": _mortality_at(48),
}
