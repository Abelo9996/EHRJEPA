"""MIMIC-IV -> canonical MEDS, by way of ``meds_etl``'s MIMIC-IV ETL.

We do not re-implement the MIMIC-IV mapping: ``meds_etl.mimic`` already encodes
the table-by-table code/time/value mapping and ships the OMOP concept maps, and
re-deriving that would be a large amount of unverifiable work. What this module
owns is (a) making the source tree look the way ``meds_etl`` expects, (b) running
it under the :mod:`ehrjepa.data.etl._meds_compat` shim, and (c) repartitioning its
output into our canonical layout with *our* deterministic splits.

``meds_etl.mimic`` insists on a ``<src>/2.2/{hosp,icu}`` layout. The PhysioNet demo
unpacks as ``mimic-iv-clinical-database-demo-2.2/{hosp,icu}``, so when the input
directory already contains ``hosp/`` we build a one-entry symlink farm pointing a
``2.2`` name at it rather than copying 100+ MB of gzipped CSV.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl

from ehrjepa.data.canonical import CANONICAL_COLUMNS, SourceExtract

from . import _meds_compat

_meds_compat.install()

MIMIC_VERSION = "2.2"

__all__ = ["extract", "from_meds_etl_output", "run_meds_etl"]


def _source_root(input_dir: Path, work_dir: Path) -> Path:
    """Return a directory containing a ``2.2`` subdirectory of MIMIC-IV tables."""
    if (input_dir / MIMIC_VERSION / "hosp").is_dir():
        return input_dir
    if not (input_dir / "hosp").is_dir():
        raise ValueError(
            f"{input_dir} looks like neither a MIMIC-IV root (with a {MIMIC_VERSION}/ "
            "subdirectory) nor an unpacked release (with a hosp/ subdirectory)"
        )
    farm = work_dir / "src"
    farm.mkdir(parents=True, exist_ok=True)
    link = farm / MIMIC_VERSION
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(input_dir.resolve(), target_is_directory=True)
    return farm


def run_meds_etl(input_dir: Path, work_dir: Path, num_shards: int = 4, num_proc: int = 1) -> Path:
    """Run ``meds_etl.mimic`` and return the directory holding its MEDS output."""
    import meds_etl.mimic

    destination = work_dir / "meds_etl_out"
    if destination.exists():
        shutil.rmtree(destination)
    source = _source_root(input_dir, work_dir)

    argv = sys.argv
    sys.argv = [
        "meds_etl_mimic",
        str(source),
        str(destination),
        "--num_shards",
        str(num_shards),
        "--num_proc",
        str(num_proc),
    ]
    try:
        meds_etl.mimic.main()
    finally:
        sys.argv = argv
    return destination


def from_meds_etl_output(meds_etl_dir: Path, source: str) -> SourceExtract:
    """Wrap a finished ``meds_etl`` MEDS directory as a :class:`SourceExtract`.

    ``meds_etl`` 0.3.2 writes the pre-0.3.2 column name ``patient_id`` and carries
    a long tail of source-specific metadata columns (``hadm_id``, ``unit``,
    ``caregiver_id``, ...). We rename the former and drop the latter: the point of
    the canonical layout is that phase 3 sees one column set regardless of source.
    The dropped columns remain available in ``meds_etl``'s own output if a later
    phase wants them.
    """
    files = sorted((meds_etl_dir / "data").glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet shards under {meds_etl_dir / 'data'}")

    def core(path: Path) -> pl.LazyFrame:
        frame = pl.scan_parquet(path)
        names = frame.collect_schema().names()
        subject = "subject_id" if "subject_id" in names else "patient_id"
        return frame.select(
            pl.col(subject).cast(pl.Int64).alias("subject_id"),
            pl.col("time").cast(pl.Datetime("us")),
            pl.col("code").cast(pl.String),
            pl.col("numeric_value").cast(pl.Float32),
            (pl.col("text_value") if "text_value" in names else pl.lit(None, dtype=pl.String))
            .cast(pl.String)
            .alias("text_value"),
        )

    events = pl.concat([core(p) for p in files], how="vertical")
    subject_ids = (
        events.select("subject_id").unique().collect(engine="streaming").to_series().sort()
    )

    codes_path = meds_etl_dir / "metadata" / "codes.parquet"
    code_metadata = None
    if codes_path.exists():
        code_metadata = pl.read_parquet(codes_path).select(
            [
                c
                for c in ("code", "description", "parent_codes")
                if c in pl.read_parquet_schema(codes_path)
            ]
        )

    return SourceExtract(
        dataset_name="MIMIC-IV",
        source=source,
        subject_ids=subject_ids,
        tables={"meds_etl": events.select(CANONICAL_COLUMNS)},
        code_metadata=code_metadata,
        notes={"meds_etl_shards": len(files)},
    )


def extract(input_dir: str | Path, work_dir: str | Path | None = None) -> SourceExtract:
    """Run ``meds_etl`` on a MIMIC-IV tree and return the resulting extract."""
    input_dir = Path(input_dir)
    temp = None
    if work_dir is None:
        temp = tempfile.mkdtemp(prefix="ehrjepa-mimic-")
        work = Path(temp)
    else:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

    meds_etl_dir = work / "meds_etl_out"
    if not (meds_etl_dir / "data").is_dir():
        meds_etl_dir = run_meds_etl(input_dir, work)
    result = from_meds_etl_output(meds_etl_dir, source=str(input_dir))
    if temp is not None:
        # The lazy frames still point at files under `work`, so materialise before
        # tearing the scratch directory down.
        result = _materialise(result, temp)
    return result


def _materialise(extract_: SourceExtract, temp: str) -> SourceExtract:
    """Collect the lazy tables so a scratch ``meds_etl`` directory can be removed."""
    tables = {name: frame.collect().lazy() for name, frame in extract_.tables.items()}
    shutil.rmtree(temp, ignore_errors=True)
    return SourceExtract(
        dataset_name=extract_.dataset_name,
        source=extract_.source,
        subject_ids=extract_.subject_ids,
        tables=tables,
        code_metadata=extract_.code_metadata,
        id_map=extract_.id_map,
        notes=extract_.notes,
    )
