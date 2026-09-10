"""PhysioNet/CinC Challenge 2012 (in-hospital mortality) -> canonical MEDS.

The challenge ships three sets of 4,000 ICU stays -- ``set-a``, ``set-b``,
``set-c`` -- as one comma-separated file per stay, plus one ``Outcomes-<set>.txt``
per set. All three outcome files were released, so all three sets are ingested;
their ``RecordID`` ranges do not overlap, so the record id is the subject id.

A stay file is long, not wide: ``Time,Parameter,Value`` with ``Time`` as
``HH:MM`` **since ICU admission**, covering the first 48 hours. Five general
descriptors (``RecordID``, ``Age``, ``Gender``, ``Height``, ``ICUType``,
``Weight``) sit at ``00:00``; everything else is one of 37 irregularly sampled
time-series channels.

Synthetic time
--------------
As with :mod:`ehrjepa.data.etl.physionet2019`, only relative time exists, so
every stay is placed on the same synthetic clock: ``HH:MM`` becomes
``2100-01-01T00:00 + HH:MM``. A MEDS timestamp is therefore readable directly as
"time since ICU admission", and the year 2100 keeps a synthetic stamp from ever
passing for a real one. ``MEDS_BIRTH`` is emitted at ``ORIGIN - Age years`` so
the cache's ``age`` feature carries the real age; the birth date itself is
fiction. Ages over 89 are reported as 90 by the challenge de-identification.

Outcome placement
-----------------
Every outcome event -- ``MEDS_DEATH``, ``LOS``, ``SAPS_I``, ``SOFA``,
``SURVIVAL`` -- is placed at :data:`OUTCOME_HOUR` (49), one hour past the end of
the 48-hour observation window, **not** at the last recorded measurement. Two
reasons. First, the record end is not the outcome time: it is wherever the
48-hour window happened to stop. Second, the mortality tasks anchor at hour 24
and hour 48, and history is everything strictly before the anchor, so an outcome
event at or before hour 48 would put the answer inside the hour-48 window.
``SAPS_I`` and ``SOFA`` are scored from the first 24 hours and would be
legitimate features, but they are severity summaries of the labelled outcome's
own drivers, so they are parked at the outcome time with everything else rather
than being handed to the model.

Code conventions
----------------
``MEDS_BIRTH``; ``MEDS_DEATH``; ``SEX//{M,F}``; ``AGE`` (numeric, years);
``UNIT//{CCU,CSRU,MICU,SICU}`` from ``ICUType``; ``VAR//<name>`` (numeric) once
per measurement, including ``VAR//Height`` and ``VAR//Weight``; ``LOS``
(numeric, days), ``SAPS_I``, ``SOFA``, ``SURVIVAL`` (numeric, days) at the
outcome time.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ehrjepa.data.canonical import SourceExtract

__all__ = [
    "DESCRIPTIONS",
    "ORIGIN",
    "OUTCOME_HOUR",
    "STATIC_PARAMETERS",
    "extract",
    "find_files",
]

#: Hour 0 of every stay: the ``00:00`` row of a record file.
ORIGIN = dt.datetime(2100, 1, 1)

#: Hours after ICU admission at which every outcome event is placed. One hour past
#: the 48-hour observation window, so it can never enter an hour-48 history.
OUTCOME_HOUR = 49

#: General descriptors, which become static events rather than ``VAR//`` events.
#: ``Height`` and ``Weight`` are deliberately *not* here: both are measurements
#: (``Weight`` is re-recorded during the stay), so they stay in the time series.
STATIC_PARAMETERS: tuple[str, ...] = ("RecordID", "Age", "Gender", "ICUType")

#: ``ICUType`` code book, from the challenge's general-descriptor table.
ICU_TYPES: dict[int, str] = {1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}

#: Parameters whose ``-1`` is the challenge's "missing" sentinel, not a value.
_SENTINEL_MINUS_ONE: tuple[str, ...] = ("Height", "Weight")

#: Time-series channels, with full names and units from the challenge description.
_VARIABLES: dict[str, str] = {
    "Albumin": "Albumin (g/dL)",
    "ALP": "Alkaline phosphatase (IU/L)",
    "ALT": "Alanine transaminase (IU/L)",
    "AST": "Aspartate transaminase (IU/L)",
    "Bilirubin": "Bilirubin (mg/dL)",
    "BUN": "Blood urea nitrogen (mg/dL)",
    "Cholesterol": "Cholesterol (mg/dL)",
    "Creatinine": "Serum creatinine (mg/dL)",
    "DiasABP": "Invasive diastolic arterial blood pressure (mmHg)",
    "FiO2": "Fractional inspired oxygen (0-1)",
    "GCS": "Glasgow Coma Score (3-15)",
    "Glucose": "Serum glucose (mg/dL)",
    "HCO3": "Serum bicarbonate (mmol/L)",
    "HCT": "Hematocrit (%)",
    "Height": "Height (cm)",
    "HR": "Heart rate (beats per minute)",
    "K": "Serum potassium (mEq/L)",
    "Lactate": "Lactate (mmol/L)",
    "MAP": "Invasive mean arterial blood pressure (mmHg)",
    "MechVent": "Mechanical ventilation respiration (1 = true)",
    "Mg": "Serum magnesium (mmol/L)",
    "Na": "Serum sodium (mEq/L)",
    "NIDiasABP": "Non-invasive diastolic arterial blood pressure (mmHg)",
    "NIMAP": "Non-invasive mean arterial blood pressure (mmHg)",
    "NISysABP": "Non-invasive systolic arterial blood pressure (mmHg)",
    "PaCO2": "Partial pressure of arterial carbon dioxide (mmHg)",
    "PaO2": "Partial pressure of arterial oxygen (mmHg)",
    "pH": "Arterial pH (0-14)",
    "Platelets": "Platelets (cells/nL)",
    "RespRate": "Respiration rate (breaths per minute)",
    "SaO2": "Oxygen saturation in hemoglobin (%)",
    "SysABP": "Invasive systolic arterial blood pressure (mmHg)",
    "Temp": "Temperature (deg C)",
    "TroponinI": "Troponin-I (ug/L)",
    "TroponinT": "Troponin-T (ug/L)",
    "Urine": "Urine output (mL)",
    "WBC": "White blood cell count (cells/nL)",
    "Weight": "Weight (kg)",
}

#: Variable full names, keyed by MEDS code.
DESCRIPTIONS: dict[str, str] = {
    **{f"VAR//{name}": text for name, text in _VARIABLES.items()},
    "AGE": "Age at ICU admission (years; 90 for patients over 89)",
    "SEX//M": "Male",
    "SEX//F": "Female",
    "UNIT//CCU": "Coronary Care Unit (ICUType 1)",
    "UNIT//CSRU": "Cardiac Surgery Recovery Unit (ICUType 2)",
    "UNIT//MICU": "Medical ICU (ICUType 3)",
    "UNIT//SICU": "Surgical ICU (ICUType 4)",
    "MEDS_BIRTH": "Synthetic birth timestamp: hour 0 of the stay minus Age years",
    "MEDS_DEATH": "In-hospital death (Outcomes column In-hospital_death)",
    "LOS": "Length of hospital stay (days)",
    "SAPS_I": "SAPS-I score from the first 24 hours",
    "SOFA": "SOFA score from the first 24 hours",
    "SURVIVAL": "Days between ICU admission and death, when known",
}

_OUTCOME_RENAMES: dict[str, str] = {
    "SAPS-I": "SAPS_I",
    "SOFA": "SOFA",
    "Length_of_stay": "LOS",
    "Survival": "SURVIVAL",
}


def find_files(input_dir: str | Path) -> dict[str, object]:
    """Locate the per-stay record files and the outcome tables under ``input_dir``.

    Returns ``{"records": [Path, ...], "outcomes": [Path, ...]}``. Record files
    are any ``<digits>.txt`` (the challenge names them after the ``RecordID``);
    ``Outcomes-*.txt`` are picked up separately and are not records.
    """
    root = Path(input_dir)
    records: list[Path] = []
    outcomes: list[Path] = []
    for path in sorted(root.rglob("*.txt")):
        if path.name.lower().startswith("outcomes"):
            outcomes.append(path)
        elif path.stem.isdigit():
            records.append(path)
    if not records:
        raise ValueError(f"no PhysioNet 2012 record files (<RecordID>.txt) found under {root}")
    if not outcomes:
        raise ValueError(f"no Outcomes-*.txt found under {root}")
    return {"records": records, "outcomes": outcomes}


def _scan_records(paths: Sequence[Path]) -> pl.LazyFrame:
    """Scan the record files into ``subject_id``/``time``/``parameter``/``value``.

    ``Time`` is ``HH:MM`` with ``HH`` running past 24, which no datetime format
    string parses, so it is split on the colon and turned into a duration.
    """
    return (
        pl.scan_csv(
            list(paths),
            schema={"Time": pl.String, "Parameter": pl.String, "Value": pl.String},
            has_header=True,
            include_file_paths="_path",
        )
        .with_columns(
            subject_id=pl.col("_path").str.extract(r"(\d+)\.txt$", 1).cast(pl.Int64, strict=False),
            _hours=pl.col("Time").str.split(":").list.get(0).cast(pl.Int64, strict=False),
            _minutes=pl.col("Time").str.split(":").list.get(1).cast(pl.Int64, strict=False),
            value=pl.col("Value").cast(pl.Float64, strict=False),
            parameter=pl.col("Parameter").str.strip_chars(),
        )
        .with_columns(
            time=(
                pl.lit(ORIGIN) + pl.duration(hours=pl.col("_hours"), minutes=pl.col("_minutes"))
            ).cast(pl.Datetime("us"))
        )
        .select("subject_id", "time", "parameter", "value")
        .drop_nulls(["subject_id", "time", "parameter"])
    )


def _measurement_events(scan: pl.LazyFrame) -> pl.LazyFrame:
    """One ``VAR//<name>`` event per measurement, sentinels and unknowns dropped."""
    known = list(_VARIABLES)
    sentinel = pl.col("parameter").is_in(list(_SENTINEL_MINUS_ONE)) & (pl.col("value") <= -1.0)
    return (
        scan.filter(pl.col("parameter").is_in(known))
        .drop_nulls("value")
        .filter(~sentinel)
        .select(
            "subject_id",
            "time",
            code=pl.lit("VAR//") + pl.col("parameter"),
            numeric_value=pl.col("value").cast(pl.Float32),
        )
    )


def _per_stay(scan: pl.LazyFrame) -> pl.DataFrame:
    """One row per stay: the general descriptors and the measurement count."""
    return (
        scan.group_by("subject_id")
        .agg(
            age=pl.col("value").filter(pl.col("parameter") == "Age").drop_nulls().first(),
            gender=pl.col("value").filter(pl.col("parameter") == "Gender").drop_nulls().first(),
            icu_type=pl.col("value").filter(pl.col("parameter") == "ICUType").drop_nulls().first(),
            n_rows=pl.len(),
        )
        .collect(engine="streaming")
        .sort("subject_id")
    )


def _static_tables(per_stay: pl.DataFrame) -> dict[str, pl.LazyFrame]:
    """Birth, age, sex and ICU unit, all stamped at hour 0."""
    frame = per_stay.lazy().with_columns(time=pl.lit(ORIGIN).cast(pl.Datetime("us")))
    unit = (
        pl.col("icu_type")
        .cast(pl.Int64, strict=False)
        .replace_strict(ICU_TYPES, default=None, return_dtype=pl.String)
    )
    return {
        "age": frame.filter(pl.col("age") > 0).select(
            "subject_id",
            "time",
            code=pl.lit("AGE"),
            numeric_value=pl.col("age").cast(pl.Float32),
        ),
        "birth": frame.filter(pl.col("age") > 0).select(
            "subject_id",
            time=(
                pl.col("time") - pl.duration(days=(pl.col("age") * 365.25).round().cast(pl.Int64))
            ).cast(pl.Datetime("us")),
            code=pl.lit("MEDS_BIRTH"),
        ),
        "sex": frame.filter(pl.col("gender").is_in([0.0, 1.0])).select(
            "subject_id",
            "time",
            code=pl.when(pl.col("gender") == 1.0)
            .then(pl.lit("SEX//M"))
            .otherwise(pl.lit("SEX//F")),
        ),
        "unit": frame.select("subject_id", "time", code=pl.lit("UNIT//") + unit).drop_nulls("code"),
    }


def read_outcomes(paths: Sequence[Path]) -> pl.DataFrame:
    """The concatenated ``Outcomes-*.txt`` tables, keyed by ``subject_id``."""
    frames = [pl.read_csv(path, infer_schema_length=0) for path in paths]
    outcomes = pl.concat(frames, how="vertical").select(
        pl.col("RecordID").cast(pl.Int64, strict=False).alias("subject_id"),
        *[
            pl.col(source).cast(pl.Float64, strict=False).alias(target)
            for source, target in _OUTCOME_RENAMES.items()
        ],
        pl.col("In-hospital_death").cast(pl.Int64, strict=False).alias("death"),
    )
    outcomes = outcomes.drop_nulls("subject_id").unique(subset=["subject_id"], keep="first")
    return outcomes.sort("subject_id")


def _outcome_tables(outcomes: pl.DataFrame) -> dict[str, pl.LazyFrame]:
    """Death plus the numeric outcome summaries, all at :data:`OUTCOME_HOUR`."""
    frame = outcomes.lazy().with_columns(
        time=(pl.lit(ORIGIN) + pl.duration(hours=OUTCOME_HOUR)).cast(pl.Datetime("us"))
    )
    tables: dict[str, pl.LazyFrame] = {
        "death": frame.filter(pl.col("death") == 1).select(
            "subject_id", "time", code=pl.lit("MEDS_DEATH")
        )
    }
    for column in ("LOS", "SAPS_I", "SOFA"):
        tables[column.lower()] = frame.drop_nulls(column).select(
            "subject_id",
            "time",
            code=pl.lit(column),
            numeric_value=pl.col(column).cast(pl.Float32),
        )
    # Survival is -1 when the survival time is not known; that is a sentinel, not
    # a duration, so those stays get no SURVIVAL event.
    tables["survival"] = frame.filter(pl.col("SURVIVAL") >= 0).select(
        "subject_id",
        "time",
        code=pl.lit("SURVIVAL"),
        numeric_value=pl.col("SURVIVAL").cast(pl.Float32),
    )
    return tables


def _code_metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {"code": list(DESCRIPTIONS), "description": list(DESCRIPTIONS.values())},
        schema={"code": pl.String, "description": pl.String},
    )


def extract(input_dir: str | Path) -> SourceExtract:
    """Build a :class:`SourceExtract` for every challenge-2012 record under ``input_dir``."""
    input_dir = Path(input_dir)
    groups = find_files(input_dir)
    records: list[Path] = groups["records"]  # type: ignore[assignment]
    outcome_paths: list[Path] = groups["outcomes"]  # type: ignore[assignment]

    scan = _scan_records(records)
    per_stay = _per_stay(scan)
    outcomes = read_outcomes(outcome_paths).join(
        per_stay.select("subject_id"), on="subject_id", how="semi"
    )

    subject_ids = pl.Series(
        "subject_id", sorted(int(path.stem) for path in records), dtype=pl.Int64
    )
    if subject_ids.n_unique() != subject_ids.len():
        raise ValueError("duplicate record ids across the challenge-2012 sets")

    tables: dict[str, pl.LazyFrame] = {"measurements": _measurement_events(scan)}
    tables.update(_static_tables(per_stay))
    tables.update(_outcome_tables(outcomes))

    return SourceExtract(
        dataset_name="PhysioNet2012",
        source=str(input_dir),
        subject_ids=subject_ids,
        tables=tables,
        code_metadata=_code_metadata(),
        notes={
            "records": len(records),
            "outcome_files": len(outcome_paths),
            "with_outcome": outcomes.height,
            "deaths": int(outcomes["death"].sum()),
            "raw_rows": int(per_stay["n_rows"].sum()),
        },
    )
