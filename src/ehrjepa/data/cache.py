"""Tensorization: a canonical MEDS directory becomes a flat, memmap-friendly cache.

The cache is one array per feature per split, concatenated across every subject in
the split, plus a ``subjects.parquet`` index that says where each subject's slice
starts and how long it is::

    <cache>/
      vocab.parquet  quantizer.parquet  vocab.json  meta.json
      train/  code_id.npy value_bin.npy value_z.npy age.npy log_delta.npy
              time_min.npy subjects.parquet
      tuning/ ...
      held_out/ ...

Flat ``.npy`` beats one file per subject (14M events is 116k files for DE-SynPUF)
and beats a single ragged parquet (which cannot be sliced without decoding). The
arrays are written through :func:`numpy.lib.format.open_memmap` one shard at a
time, so the writer's memory is bounded by the largest input shard, and read back
with ``mmap_mode="r"``, so a training run pages in only the windows it samples.

Feature semantics, per event:

``code_id``    vocabulary id, after hierarchical fallback (see :mod:`~.tokenize`).
``value_bin``  0 when the event carries no number, else the 1..10 decile bin. A
               value sitting exactly on an edge falls in the lower bin, which is
               what keeps a code whose values are mostly one constant (very
               common: dose counts, "days supplied") from smearing across bins.
``value_z``    per-code z-score of the (optionally ``log1p``-ed) value, clipped
               to +-5, and 0 when there is no number.
``age``        years since ``MEDS_BIRTH``, clipped to [0, 120]. Subjects with no
               birth event are anchored at their first event and flagged
               ``has_birth = false`` in the subject index.
``log_delta``  ``log1p`` of the hours since the subject's previous event; 0 for
               the first event and for ties, which keep their parquet order.
``time_min``   raw event time as int64 minutes since epoch, kept so that
               downstream label windows can be aligned without re-reading MEDS.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import meds
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ehrjepa.data.canonical import SPLITS
from ehrjepa.data.tokenize import (
    DEFAULT_MIN_COUNT,
    DEFAULT_MIN_VALUE_OBS,
    N_VALUE_BINS,
    TOKENIZER_VERSION,
    Z_CLIP,
    Vocabulary,
    fit_tokenizer,
    split_shards,
    write_fit,
)

__all__ = [
    "FEATURES",
    "SUBJECT_INDEX_COLUMNS",
    "build_cache",
    "read_meta",
    "shard_features",
    "tensorize_split",
]

#: Feature name -> on-disk dtype. The order is the order in ``meta.json``.
FEATURES: Mapping[str, np.dtype] = {
    "code_id": np.dtype(np.int32),
    "value_bin": np.dtype(np.int8),
    "value_z": np.dtype(np.float32),
    "age": np.dtype(np.float32),
    "log_delta": np.dtype(np.float32),
    "time_min": np.dtype(np.int64),
}

SUBJECT_INDEX_COLUMNS: tuple[str, ...] = ("subject_id", "offset", "length", "split", "has_birth")

_SECONDS_PER_YEAR = 365.25 * 24 * 3600
_MAX_AGE_YEARS = 120.0
_EDGE_COLUMNS: tuple[str, ...] = tuple(f"edge_{i}" for i in range(N_VALUE_BINS - 1))
_KIND_NAMES = ("direct", "ancestor", "unk")


def _mapping_frame(
    codes: Sequence[str], vocab: Vocabulary, cache: dict[str, tuple[int, int]]
) -> pl.DataFrame:
    """``code -> (code_id, kind)`` for ``codes``, memoised across shards."""
    kinds = {"direct": 0, "ancestor": 1, "unk": 2}
    for code in codes:
        if code not in cache:
            code_id, kind = vocab.resolve(code)
            cache[code] = (code_id, kinds[kind])
    return pl.DataFrame(
        {
            "code": pl.Series(list(codes), dtype=pl.String),
            "code_id": pl.Series([cache[c][0] for c in codes], dtype=pl.Int32),
            "kind": pl.Series([cache[c][1] for c in codes], dtype=pl.Int8),
        }
    )


def shard_features(
    events: pl.DataFrame,
    vocab: Vocabulary,
    quantizer: pl.DataFrame,
    resolver_cache: dict[str, tuple[int, int]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Turn one MEDS shard into per-event features plus a per-subject index.

    ``events`` must already be in canonical order -- sorted by
    ``(subject_id, time, code)``, subjects contiguous -- which is what
    :func:`ehrjepa.data.canonical.write_canonical` guarantees. The row order of
    the returned feature frame is the row order of ``events``.
    """
    cache = {} if resolver_cache is None else resolver_cache
    mapping = _mapping_frame(events["code"].unique().to_list(), vocab, cache)

    anchors = events.group_by("subject_id").agg(
        birth=pl.col("time").filter(pl.col("code") == meds.birth_code).min(),
        first=pl.col("time").min(),
    )
    anchors = anchors.with_columns(
        has_birth=pl.col("birth").is_not_null(),
        anchor=pl.coalesce("birth", "first"),
    ).select("subject_id", "has_birth", "anchor")

    quant = quantizer.select("code_id", "use_log1p", "mean", "std", *_EDGE_COLUMNS)
    frame = (
        events.with_row_index("_row")
        .join(mapping, on="code", how="left")
        .join(anchors, on="subject_id", how="left")
        .join(quant, on="code_id", how="left")
        .sort("_row")
    )

    value = pl.col("numeric_value").cast(pl.Float64)
    present = value.is_not_null() & value.is_not_nan()
    # NaN edges (a code whose quantizer never saw a value) compare false, so an
    # unquantised but present value lands in bin 1 rather than going missing.
    bin_index = pl.sum_horizontal([(value > pl.col(e)).cast(pl.Int32) for e in _EDGE_COLUMNS]) + 1
    transformed = (
        pl.when(pl.col("use_log1p")).then(value.clip(lower_bound=0.0).log1p()).otherwise(value)
    )
    event_time = pl.coalesce("time", "anchor")
    delta_hours = (event_time - event_time.shift(1).over("subject_id")).dt.total_seconds() / 3600.0

    features = frame.select(
        pl.col("subject_id"),
        pl.col("code_id").fill_null(1).cast(pl.Int32),
        pl.col("kind").fill_null(2).cast(pl.Int8),
        value_bin=pl.when(present).then(bin_index.clip(1, N_VALUE_BINS)).otherwise(0).cast(pl.Int8),
        value_z=pl.when(present & (pl.col("std") > 0))
        .then(((transformed - pl.col("mean")) / pl.col("std")).clip(-Z_CLIP, Z_CLIP))
        .otherwise(0.0)
        .fill_nan(0.0)
        .cast(pl.Float32),
        age=((event_time - pl.col("anchor")).dt.total_seconds() / _SECONDS_PER_YEAR)
        .clip(0.0, _MAX_AGE_YEARS)
        .fill_null(0.0)
        .cast(pl.Float32),
        log_delta=delta_hours.clip(lower_bound=0.0).log1p().fill_null(0.0).cast(pl.Float32),
        time_min=(event_time.dt.epoch(time_unit="us") // 60_000_000).cast(pl.Int64),
    )

    lengths = features.group_by("subject_id", maintain_order=True).agg(
        length=pl.len().cast(pl.Int64)
    )
    index = lengths.join(anchors.select("subject_id", "has_birth"), on="subject_id", how="left")
    index = index.with_columns(
        offset=pl.col("length").cum_sum().shift(1, fill_value=0).cast(pl.Int64)
    )
    return features, index.select("subject_id", "offset", "length", "has_birth")


def tensorize_split(
    meds_dir: str | Path,
    cache_dir: str | Path,
    split: str,
    vocab: Vocabulary,
    quantizer: pl.DataFrame,
    resolver_cache: dict[str, tuple[int, int]] | None = None,
) -> dict[str, object]:
    """Write ``<cache_dir>/<split>/`` and return the split's fit statistics."""
    shards = split_shards(meds_dir, split)
    out = Path(cache_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    total = sum(pq.ParquetFile(shard).metadata.num_rows for shard in shards)

    arrays = {
        name: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dtype, shape=(total,))
        for name, dtype in FEATURES.items()
    }
    index_parts: list[pl.DataFrame] = []
    kind_counts = dict.fromkeys(range(3), 0)
    offset = 0
    for shard in shards:
        events = pl.read_parquet(shard)
        features, index = shard_features(events, vocab, quantizer, resolver_cache)
        n = features.height
        for name, array in arrays.items():
            array[offset : offset + n] = features[name].to_numpy()
        for kind, count in features.group_by("kind").len().iter_rows():
            kind_counts[int(kind)] += int(count)
        index_parts.append(index.with_columns(offset=pl.col("offset") + offset))
        offset += n
    for array in arrays.values():
        array.flush()
    del arrays

    empty = pl.DataFrame(
        schema={
            "subject_id": pl.Int64,
            "offset": pl.Int64,
            "length": pl.Int64,
            "has_birth": pl.Boolean,
        }
    )
    index = pl.concat(index_parts) if index_parts else empty
    index = index.with_columns(split=pl.lit(split, dtype=pl.String)).select(SUBJECT_INDEX_COLUMNS)
    index.write_parquet(out / "subjects.parquet")

    events_total = max(offset, 1)
    return {
        "events": offset,
        "subjects": index.height,
        "subjects_without_birth": int(index.height - int(index["has_birth"].sum() or 0)),
        "shards": len(shards),
        **{f"{name}_rate": kind_counts[i] / events_total for i, name in enumerate(_KIND_NAMES)},
    }


def build_cache(
    meds_dir: str | Path,
    cache_dir: str | Path,
    min_count: int = DEFAULT_MIN_COUNT,
    min_value_obs: int = DEFAULT_MIN_VALUE_OBS,
    splits: Sequence[str] = SPLITS,
) -> dict[str, object]:
    """Fit the tokenizer on train, tensorize every split, and write ``meta.json``."""
    started = time.perf_counter()
    root = Path(meds_dir)
    out = Path(cache_dir)
    vocab, quantizer, fit_stats = fit_tokenizer(
        root, min_count=min_count, min_value_obs=min_value_obs
    )
    write_fit(out, vocab, quantizer, fit_stats)

    resolver_cache: dict[str, tuple[int, int]] = {}
    per_split = {
        split: tensorize_split(root, out, split, vocab, quantizer, resolver_cache)
        for split in splits
    }

    dataset_path = root / meds.dataset_metadata_filepath
    source = json.loads(dataset_path.read_text()) if dataset_path.exists() else {}
    meta: dict[str, object] = {
        "tokenizer_version": TOKENIZER_VERSION,
        "source_meds_dir": str(root),
        "source_dataset": source,
        "splits": list(splits),
        "features": {name: dtype.name for name, dtype in FEATURES.items()},
        "vocab_size": len(vocab),
        "n_value_bins": N_VALUE_BINS,
        "z_clip": Z_CLIP,
        "fit": fit_stats,
        "per_split": per_split,
        "cache_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


def read_meta(cache_dir: str | Path) -> dict[str, object]:
    """Read ``meta.json`` from a built cache."""
    return json.loads((Path(cache_dir) / "meta.json").read_text())
