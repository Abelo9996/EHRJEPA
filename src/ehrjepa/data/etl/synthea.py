"""Synthea (Coherent and vanilla CSV exports) -> canonical MEDS.

Synthea exports one CSV per domain with UUID patient ids and ISO-8601 timestamps
(``2001-04-22T08:57:56Z`` in the event tables, bare ``2001-04-22`` in a few date
columns, so both formats are tried). Only ``patients.csv`` is required; every other
table is optional and simply contributes no events when absent. The Coherent
release, in particular, ships no ``observations.csv``, so the LOINC path below is
exercised by vanilla Synthea exports and by the test fixtures.

Code conventions
----------------
``MEDS_BIRTH`` / ``MEDS_DEATH``; ``SEX//<GENDER>``, ``RACE//<race>``,
``ETHNICITY//<ethnicity>`` dated at birth; ``ENCOUNTER//<CLASS>`` at encounter
start and ``END//ENCOUNTER`` at stop; ``SNOMED//<code>`` for conditions (plus
``CONDITION_STOP//<code>`` at resolution when the row has a stop date);
``SNOMED_PROC//<code>`` for procedures; ``RXNORM//<code>`` valued with
``DISPENSES``; and ``LOINC//<code>//<unit>`` for observations, with the unit folded
into the code the way MEDS_transforms does so that a code's numeric values are
always on one scale. Unit-less observations get ``//UNK``, matching the
MEDS_transforms convention.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ehrjepa.data.canonical import SourceExtract, stable_subject_id

__all__ = ["extract"]

_TIMESTAMP_FORMATS = ("%Y-%m-%dT%H:%M:%S%.fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d")


def _scan(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, infer_schema_length=0)


def _time(column: str) -> pl.Expr:
    """Parse a Synthea timestamp, trying each format it is known to emit."""
    return pl.coalesce(
        [
            pl.col(column).str.strptime(pl.Datetime("us"), fmt, strict=False)
            for fmt in _TIMESTAMP_FORMATS
        ]
    )


def _clean(expr: pl.Expr) -> pl.Expr:
    trimmed = expr.str.strip_chars()
    return pl.when(trimmed.str.len_chars() > 0).then(trimmed).otherwise(None)


def _id_map(patients: Path) -> pl.DataFrame:
    ids = pl.read_csv(patients, infer_schema_length=0).get_column("Id").drop_nulls().unique().sort()
    mapped = pl.DataFrame(
        {
            "source_id": ids,
            "subject_id": pl.Series([stable_subject_id(str(v)) for v in ids], dtype=pl.Int64),
        }
    )
    if mapped["subject_id"].n_unique() != mapped.height:
        raise RuntimeError("subject id hash collision in Synthea id map")
    return mapped


def _join(frame: pl.LazyFrame, id_map: pl.LazyFrame, column: str) -> pl.LazyFrame:
    return frame.join(id_map, left_on=column, right_on="source_id", how="inner")


def _patient_events(path: Path, id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    patients = _join(_scan(path), id_map, "Id").with_columns(
        _time("BIRTHDATE").alias("_birth"), _time("DEATHDATE").alias("_death")
    )
    tables = {
        "birth": patients.drop_nulls("_birth").select(
            "subject_id", pl.col("_birth").alias("time"), pl.lit("MEDS_BIRTH").alias("code")
        ),
        "death": patients.drop_nulls("_death").select(
            "subject_id", pl.col("_death").alias("time"), pl.lit("MEDS_DEATH").alias("code")
        ),
    }
    for column, prefix in (("GENDER", "SEX"), ("RACE", "RACE"), ("ETHNICITY", "ETHNICITY")):
        if column not in patients.collect_schema().names():
            continue
        tables[prefix.lower()] = (
            patients.with_columns(_clean(pl.col(column)).str.to_uppercase().alias("_v"))
            .drop_nulls("_v")
            .select(
                "subject_id",
                pl.col("_birth").alias("time"),
                (pl.lit(f"{prefix}//") + pl.col("_v")).alias("code"),
            )
        )
    return tables


def _encounter_events(path: Path, id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    encounters = _join(_scan(path), id_map, "PATIENT").with_columns(
        _time("START").alias("_start"),
        _time("STOP").alias("_stop"),
        _clean(pl.col("ENCOUNTERCLASS")).str.to_uppercase().alias("_class"),
    )
    return {
        "encounter_start": encounters.drop_nulls(["_start", "_class"]).select(
            "subject_id",
            pl.col("_start").alias("time"),
            (pl.lit("ENCOUNTER//") + pl.col("_class")).alias("code"),
        ),
        "encounter_end": encounters.drop_nulls("_stop").select(
            "subject_id", pl.col("_stop").alias("time"), pl.lit("END//ENCOUNTER").alias("code")
        ),
    }


def _condition_events(path: Path, id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    conditions = _join(_scan(path), id_map, "PATIENT").with_columns(
        _time("START").alias("_start"),
        _time("STOP").alias("_stop"),
        _clean(pl.col("CODE")).alias("_code"),
    )
    return {
        "condition_start": conditions.drop_nulls(["_start", "_code"]).select(
            "subject_id",
            pl.col("_start").alias("time"),
            (pl.lit("SNOMED//") + pl.col("_code")).alias("code"),
        ),
        "condition_stop": conditions.drop_nulls(["_stop", "_code"]).select(
            "subject_id",
            pl.col("_stop").alias("time"),
            (pl.lit("CONDITION_STOP//") + pl.col("_code")).alias("code"),
        ),
    }


def _procedure_events(path: Path, id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    frame = _join(_scan(path), id_map, "PATIENT")
    names = frame.collect_schema().names()
    # Synthea renamed the procedure timestamp column from DATE to START.
    time_column = "START" if "START" in names else "DATE"
    return {
        "procedure": frame.with_columns(
            _time(time_column).alias("_time"), _clean(pl.col("CODE")).alias("_code")
        )
        .drop_nulls(["_time", "_code"])
        .select(
            "subject_id",
            pl.col("_time").alias("time"),
            (pl.lit("SNOMED_PROC//") + pl.col("_code")).alias("code"),
        )
    }


def _medication_events(path: Path, id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    frame = _join(_scan(path), id_map, "PATIENT")
    has_dispenses = "DISPENSES" in frame.collect_schema().names()
    dispenses = (
        pl.col("DISPENSES").cast(pl.Float32, strict=False)
        if has_dispenses
        else pl.lit(None, dtype=pl.Float32)
    )
    return {
        "medication": frame.with_columns(
            _time("START").alias("_time"), _clean(pl.col("CODE")).alias("_code")
        )
        .drop_nulls(["_time", "_code"])
        .select(
            "subject_id",
            pl.col("_time").alias("time"),
            (pl.lit("RXNORM//") + pl.col("_code")).alias("code"),
            dispenses.alias("numeric_value"),
        )
    }


def _observation_events(path: Path, id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    frame = _join(_scan(path), id_map, "PATIENT")
    names = frame.collect_schema().names()
    unit = _clean(pl.col("UNITS")) if "UNITS" in names else pl.lit(None, dtype=pl.String)
    numeric = pl.col("VALUE").str.strip_chars().cast(pl.Float64, strict=False)
    return {
        "observation": frame.with_columns(
            _time("DATE").alias("_time"),
            _clean(pl.col("CODE")).alias("_code"),
            unit.fill_null("UNK").str.replace_all("//", "_").alias("_unit"),
            numeric.alias("_numeric"),
        )
        .drop_nulls(["_time", "_code"])
        .select(
            "subject_id",
            pl.col("_time").alias("time"),
            (pl.lit("LOINC//") + pl.col("_code") + pl.lit("//") + pl.col("_unit")).alias("code"),
            pl.col("_numeric").cast(pl.Float32).alias("numeric_value"),
            pl.when(pl.col("_numeric").is_null())
            .then(_clean(pl.col("VALUE")))
            .otherwise(None)
            .alias("text_value"),
        )
    }


_OPTIONAL_TABLES = {
    "encounters.csv": _encounter_events,
    "conditions.csv": _condition_events,
    "procedures.csv": _procedure_events,
    "medications.csv": _medication_events,
    "observations.csv": _observation_events,
}


def extract(input_dir: str | Path) -> SourceExtract:
    """Build a :class:`SourceExtract` from a Synthea CSV export directory."""
    root = Path(input_dir)
    patients = root / "patients.csv"
    if not patients.exists():
        raise ValueError(f"{root} does not contain patients.csv")

    id_map = _id_map(patients)
    id_map_lazy = id_map.lazy()

    tables: dict[str, pl.LazyFrame] = dict(_patient_events(patients, id_map_lazy))
    present: dict[str, int] = {}
    for filename, builder in _OPTIONAL_TABLES.items():
        path = root / filename
        if path.exists():
            tables.update(builder(path, id_map_lazy))
            present[filename] = 1

    missing = sorted(set(_OPTIONAL_TABLES) - set(present))
    return SourceExtract(
        dataset_name="Synthea",
        source=str(root),
        subject_ids=id_map["subject_id"],
        tables=tables,
        id_map=id_map,
        notes={f"missing_{name.removesuffix('.csv')}": 1 for name in missing},
    )
