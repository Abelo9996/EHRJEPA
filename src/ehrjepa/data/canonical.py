"""Canonical MEDS layout: deterministic splits, sharding, validation and writing.

Every source ETL in :mod:`ehrjepa.data.etl` lowers its raw tables into a
:class:`SourceExtract` -- a bag of lazy event frames plus the subject inventory --
and hands it to :func:`write_canonical`, which is the *only* code in the project
that decides how a MEDS dataset is laid out on disk. The layout it produces is
described in ``src/ehrjepa/data/README.md``.

Splits are a pure function of ``subject_id``: ``blake2b`` of a seeded string form
of the id, taken modulo 10,000 and cut at 8000/9000. That makes them stable across
machines, polars versions and re-runs, and patient-disjoint by construction.
Shard assignment uses the same construction with a different seed prefix.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import meds
import polars as pl
import pyarrow.parquet as pq

from ehrjepa import __version__ as ETL_VERSION

__all__ = [
    "CANONICAL_COLUMNS",
    "DEFAULT_SHARD_SIZE",
    "SPLITS",
    "SPLIT_SEED",
    "SourceExtract",
    "coerce_events",
    "shard_for",
    "split_for",
    "stable_subject_id",
    "write_canonical",
]

#: The MEDS splits, in the order we always iterate them.
SPLITS: tuple[str, str, str] = (meds.train_split, meds.tuning_split, meds.held_out_split)

#: Fixed seed for the split hash. Changing this reshuffles every dataset, so don't.
SPLIT_SEED = 20240917

#: Cut points, in units of 1/10000, for train / tuning / held_out (80/10/10).
_SPLIT_CUTS: tuple[tuple[str, int], ...] = (
    (meds.train_split, 8000),
    (meds.tuning_split, 9000),
    (meds.held_out_split, 10000),
)

#: Default number of *subjects* per output shard (shards are subject-disjoint).
DEFAULT_SHARD_SIZE = 10_000

#: MEDS 0.4 core data columns, in canonical order.
CANONICAL_COLUMNS: tuple[str, ...] = ("subject_id", "time", "code", "numeric_value", "text_value")

_CANONICAL_DTYPES: Mapping[str, pl.DataType] = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
    "text_value": pl.String,
}

_SHARD_KEY = "_shard_key"


def _u64(payload: str) -> int:
    """A stable 64-bit unsigned hash of ``payload``.

    ``hashlib`` is used rather than :meth:`polars.Expr.hash` because polars makes
    no cross-version stability guarantee for its hash, and split determinism is a
    hard requirement here.
    """
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")


def stable_subject_id(source_id: str) -> int:
    """Map an opaque source identifier (a UUID, a DESYNPUF_ID) to a positive Int64.

    The top bit is dropped so the result is always a non-negative ``Int64``. With
    63 bits of range the collision probability is negligible at the scales we care
    about (~1e-9 for a few million subjects), and every ETL asserts uniqueness of
    the mapping it builds anyway.
    """
    return _u64(f"subject:{source_id}") >> 1


def split_for(subject_id: int, seed: int = SPLIT_SEED) -> str:
    """Return the MEDS split for ``subject_id``. Deterministic and 80/10/10."""
    bucket = _u64(f"split:{seed}:{subject_id}") % 10_000
    for name, cut in _SPLIT_CUTS:
        if bucket < cut:
            return name
    raise AssertionError("unreachable")  # pragma: no cover


def shard_for(subject_id: int, n_shards: int, seed: int = SPLIT_SEED) -> int:
    """Return the shard index within a split for ``subject_id``."""
    if n_shards <= 1:
        return 0
    return _u64(f"shard:{seed}:{subject_id}") % n_shards


@dataclass
class SourceExtract:
    """What a source ETL produces, before it is laid out on disk.

    Attributes:
        dataset_name: Human name written into ``metadata/dataset.json``.
        source: Provenance string (usually the input path or a dataset release id).
        subject_ids: Every subject in the dataset, as an ``Int64`` series. Drives
            split assignment, so it must be the complete inventory even for
            subjects that end up with zero events.
        tables: Named lazy frames of events. Each must expose at least
            ``subject_id``/``time``/``code``; ``numeric_value``/``text_value`` are
            filled with nulls when absent. Frames are consumed exactly once, by a
            streaming sink, so they may be arbitrarily large.
        code_metadata: Optional ``code``/``description``/``parent_codes`` frame.
            Codes observed in the data but missing here get a null description.
        id_map: Optional ``source_id``/``subject_id`` frame, written to
            ``metadata/subject_id_map.parquet`` when the source uses opaque ids.
        notes: Free-form counters surfaced by the CLI (dropped rows, and so on).
    """

    dataset_name: str
    source: str
    subject_ids: pl.Series
    tables: Mapping[str, pl.LazyFrame]
    code_metadata: pl.DataFrame | None = None
    id_map: pl.DataFrame | None = None
    notes: dict[str, int] = field(default_factory=dict)


def coerce_events(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Project ``frame`` onto the canonical MEDS columns with canonical dtypes.

    Missing optional columns are materialised as nulls, extra columns are dropped,
    and rows with a null ``subject_id`` or ``code`` are discarded -- MEDS forbids
    both, and every source has at least some rows that hit one of them.
    """
    names = set(frame.collect_schema().names())
    missing = {"subject_id", "time", "code"} - names
    if missing:
        raise ValueError(f"event frame is missing required column(s): {sorted(missing)}")

    exprs = []
    for column, dtype in _CANONICAL_DTYPES.items():
        if column in names:
            exprs.append(pl.col(column).cast(dtype).alias(column))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(column))
    return (
        frame.select(exprs)
        .filter(pl.col("subject_id").is_not_null() & pl.col("code").is_not_null())
        .filter(pl.col("code").str.len_chars() > 0)
    )


def assign_subjects(
    subject_ids: pl.Series, shard_size: int = DEFAULT_SHARD_SIZE, seed: int = SPLIT_SEED
) -> pl.DataFrame:
    """Build the ``subject_id`` -> ``(split, shard)`` assignment table.

    The number of shards is computed *per split* so that no shard holds more than
    ``shard_size`` subjects; within a split, subjects are spread by an independent
    hash rather than by rank, which keeps the assignment a pure function of the id.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")

    ids = subject_ids.cast(pl.Int64).unique().sort()
    splits = [split_for(int(sid), seed) for sid in ids]
    counts = {name: splits.count(name) for name in SPLITS}
    n_shards = {name: max(1, math.ceil(counts[name] / shard_size)) for name in SPLITS}
    shards = [
        shard_for(int(sid), n_shards[split], seed) for sid, split in zip(ids, splits, strict=True)
    ]

    return pl.DataFrame(
        {
            "subject_id": ids,
            "split": pl.Series(splits, dtype=pl.String),
            "shard": pl.Series(shards, dtype=pl.Int32),
        }
    ).with_columns(
        pl.format("{}-{}", pl.col("split"), pl.col("shard").cast(pl.String).str.zfill(5)).alias(
            _SHARD_KEY
        )
    )


def _validated_arrow(frame: pl.DataFrame):
    """Align a shard to :class:`meds.DataSchema` and validate it."""
    table = meds.DataSchema.align(frame.to_arrow())
    meds.DataSchema.validate(table)
    return table


def write_canonical(
    extract: SourceExtract,
    out_dir: str | Path,
    shard_size: int = DEFAULT_SHARD_SIZE,
    seed: int = SPLIT_SEED,
    overwrite: bool = True,
) -> dict[str, object]:
    """Write ``extract`` to ``out_dir`` in the canonical MEDS 0.4 layout.

    The write is two-phase. Phase one streams every source table through a
    partitioned parquet sink keyed by ``(split, shard)``, so the raw event stream
    is never held in memory. Phase two reads each partition back -- one shard's
    worth of subjects, bounded by ``shard_size`` -- sorts it, validates it against
    :class:`meds.DataSchema` and writes the final file.

    Returns a summary dict (event and subject counts, per-split counts, shard
    files) for the caller to log.
    """
    out = Path(out_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} already exists")
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)

    assignment = assign_subjects(extract.subject_ids, shard_size=shard_size, seed=seed)
    assignment.select("subject_id", "split").pipe(
        lambda df: pq.write_table(
            meds.SubjectSplitSchema.align(df.to_arrow()), out / meds.subject_splits_filepath
        )
    )
    if extract.id_map is not None:
        extract.id_map.write_parquet(out / "metadata" / "subject_id_map.parquet")

    staging = out / ".staging"
    key_lookup = assignment.lazy().select("subject_id", _SHARD_KEY)
    for name, frame in extract.tables.items():
        (
            coerce_events(frame)
            .join(key_lookup, on="subject_id", how="inner")
            .sink_parquet(pl.PartitionBy(staging / name, key=_SHARD_KEY, include_key=False))
        )

    observed_codes: list[pl.DataFrame] = []
    n_events = 0
    shard_files: list[str] = []
    split_events = dict.fromkeys(SPLITS, 0)

    for row in (
        assignment.select("split", "shard", _SHARD_KEY).unique().sort("split", "shard").iter_rows()
    ):
        split, shard, key = row
        parts = sorted(staging.glob(f"*/{_SHARD_KEY}={key}/*.parquet"))
        if not parts:
            continue
        shard_frame = (
            pl.concat([pl.scan_parquet(p) for p in parts])
            .select(CANONICAL_COLUMNS)
            .sort("subject_id", "time", "code", nulls_last=False)
            .collect()
        )
        (out / "data" / split).mkdir(parents=True, exist_ok=True)
        target = out / "data" / split / f"{shard}.parquet"
        pq.write_table(_validated_arrow(shard_frame), target)

        n_events += shard_frame.height
        split_events[split] += shard_frame.height
        shard_files.append(str(target.relative_to(out)))
        observed_codes.append(shard_frame.select(pl.col("code").unique()))

    for split in SPLITS:
        (out / "data" / split).mkdir(parents=True, exist_ok=True)

    _write_code_metadata(out, observed_codes, extract.code_metadata)
    _write_dataset_metadata(out, extract)
    shutil.rmtree(staging, ignore_errors=True)

    split_subjects = (
        assignment.group_by("split").len().to_dict(as_series=False)
        if assignment.height
        else {"split": [], "len": []}
    )
    return {
        "dataset_name": extract.dataset_name,
        "subjects": assignment.height,
        "events": n_events,
        "split_subjects": dict(zip(split_subjects["split"], split_subjects["len"], strict=True)),
        "split_events": split_events,
        "shards": shard_files,
        "notes": dict(extract.notes),
    }


def _write_code_metadata(
    out: Path, observed: list[pl.DataFrame], supplied: pl.DataFrame | None
) -> None:
    """Union the observed vocabulary with any source-supplied descriptions."""
    if observed:
        codes = pl.concat(observed).unique().sort("code")
    else:
        codes = pl.DataFrame({"code": pl.Series([], dtype=pl.String)})

    if supplied is not None and supplied.height:
        extra = supplied.select(
            pl.col("code").cast(pl.String),
            pl.col("description").cast(pl.String)
            if "description" in supplied.columns
            else pl.lit(None, dtype=pl.String).alias("description"),
            pl.col("parent_codes").cast(pl.List(pl.String))
            if "parent_codes" in supplied.columns
            else pl.lit(None, dtype=pl.List(pl.String)).alias("parent_codes"),
        ).unique(subset=["code"], keep="first")
        codes = codes.join(extra, on="code", how="left")
    else:
        codes = codes.with_columns(
            pl.lit(None, dtype=pl.String).alias("description"),
            pl.lit(None, dtype=pl.List(pl.String)).alias("parent_codes"),
        )

    table = meds.CodeMetadataSchema.align(
        codes.select("code", "description", "parent_codes").to_arrow()
    )
    meds.CodeMetadataSchema.validate(table)
    pq.write_table(table, out / meds.code_metadata_filepath)


def _write_dataset_metadata(out: Path, extract: SourceExtract) -> None:
    metadata = {
        "dataset_name": extract.dataset_name,
        "etl_name": "ehrjepa.data.etl",
        "etl_version": ETL_VERSION,
        "meds_version": meds.__version__,
        "source": extract.source,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    meds.DatasetMetadataSchema.validate(metadata)
    (out / meds.dataset_metadata_filepath).write_text(json.dumps(metadata, indent=2) + "\n")
