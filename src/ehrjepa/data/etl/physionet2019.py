"""PhysioNet/CinC Challenge 2019 (early sepsis prediction) -> canonical MEDS.

The challenge ships one pipe-separated file per ICU stay under
``training/training_setA`` (hospital A, 20,336 stays) and
``training/training_setB`` (hospital B, 20,000 stays). Every file has 41 columns
and one row per **hour** of the stay: 8 vital signs, 26 laboratory values, five
static descriptors (``Age``, ``Gender``, ``Unit1``, ``Unit2``, ``HospAdmTime``),
the 1-based hour index ``ICULOS``, and the challenge's ``SepsisLabel``. Missing
cells are the literal string ``NaN``.

Synthetic time
--------------
The release carries no absolute timestamps -- only ``ICULOS``, the hour of the
stay -- so every stay is placed on the same synthetic clock: ICU hour
``ICULOS = h`` becomes ``2100-01-01T00:00 + (h - 1) hours``. :data:`ORIGIN` is
therefore hour 0 of every stay, and a MEDS timestamp is readable directly as
"hours since ICU admission". The year 2100 is deliberately absurd, so a synthetic
timestamp can never be mistaken for a real one, and it keeps every stay's clock
identical, which is the only thing the relative-time features in the tensor cache
(``log_delta``) actually see.

``MEDS_BIRTH`` is emitted at ``ORIGIN - Age years`` so the cache's ``age``
feature carries the patient's real age instead of being anchored at the first
event. The birth *date* is fiction; the age it encodes is not.

Code conventions
----------------
``MEDS_BIRTH``; ``SEX//{M,F}``; ``AGE`` (numeric, years); ``UNIT//{MICU,SICU}``
from the ``Unit1``/``Unit2`` flags; ``HOSP_ADM_TIME`` (numeric, hours between
hospital admission and ICU admission, negative); ``ICU_HOUR`` (numeric,
``ICULOS``) once per recorded hour; ``VAR//<name>`` (numeric) once per non-null
measurement; ``SEPSIS_ONSET`` once per septic stay.

Sepsis labelling
----------------
``SepsisLabel`` is *already shifted*: the challenge sets it to 1 for every hour
``t >= t_sepsis - 6``, where ``t_sepsis`` is the Sepsis-3 onset time. This ETL
emits the onset itself -- a single ``SEPSIS_ONSET`` event at
``t_sepsis = first hour with SepsisLabel == 1, plus 6 hours`` -- and does **not**
emit the hourly flag. Emitting the hourly flag would put a label event six hours
*before* onset into the event stream, and every history window anchored in the
six hours leading up to onset (exactly the windows the early-prediction task is
built from) would then contain the answer. One event at the true onset time
cannot leak, because every anchor the tasks in :mod:`ehrjepa.eval.icu_tasks`
admit is strictly before it.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ehrjepa.data.canonical import SourceExtract

__all__ = ["DESCRIPTIONS", "MEASUREMENTS", "ORIGIN", "SEPSIS_SHIFT_HOURS", "extract", "find_files"]

#: Hour 0 of every stay. ``ICULOS = 1`` is the first recorded hour, so it lands here.
ORIGIN = dt.datetime(2100, 1, 1)

#: ``SepsisLabel`` is 1 from ``t_sepsis - 6`` onwards, so onset is the first flagged
#: hour plus this many hours. Fixed by the challenge's own definition.
SEPSIS_SHIFT_HOURS = 6

#: Vital signs, in the column order the PSV uses.
VITALS: tuple[str, ...] = ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2")

#: Laboratory values, in the column order the PSV uses.
LABS: tuple[str, ...] = (
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
)

#: Every channel that becomes a ``VAR//<name>`` event.
MEASUREMENTS: tuple[str, ...] = (*VITALS, *LABS)

#: Static descriptors and bookkeeping columns, in PSV order.
STATIC_COLUMNS: tuple[str, ...] = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")

_BOOKKEEPING: tuple[str, ...] = ("ICULOS", "SepsisLabel")

#: Every PSV column, in order. Pinned so the schema does not depend on which file
#: polars happens to sniff first out of forty thousand.
PSV_COLUMNS: tuple[str, ...] = (*MEASUREMENTS, *STATIC_COLUMNS, *_BOOKKEEPING)

#: Variable full names and units, transcribed from the challenge's data description.
DESCRIPTIONS: dict[str, str] = {
    "VAR//HR": "Heart rate (beats per minute)",
    "VAR//O2Sat": "Pulse oximetry (%)",
    "VAR//Temp": "Temperature (deg C)",
    "VAR//SBP": "Systolic blood pressure (mm Hg)",
    "VAR//MAP": "Mean arterial pressure (mm Hg)",
    "VAR//DBP": "Diastolic blood pressure (mm Hg)",
    "VAR//Resp": "Respiration rate (breaths per minute)",
    "VAR//EtCO2": "End tidal carbon dioxide (mm Hg)",
    "VAR//BaseExcess": "Excess bicarbonate (mmol/L)",
    "VAR//HCO3": "Bicarbonate (mmol/L)",
    "VAR//FiO2": "Fraction of inspired oxygen (%)",
    "VAR//pH": "Arterial pH (unitless)",
    "VAR//PaCO2": "Partial pressure of carbon dioxide from arterial blood (mm Hg)",
    "VAR//SaO2": "Oxygen saturation from arterial blood (%)",
    "VAR//AST": "Aspartate transaminase (IU/L)",
    "VAR//BUN": "Blood urea nitrogen (mg/dL)",
    "VAR//Alkalinephos": "Alkaline phosphatase (IU/L)",
    "VAR//Calcium": "Calcium (mg/dL)",
    "VAR//Chloride": "Chloride (mmol/L)",
    "VAR//Creatinine": "Creatinine (mg/dL)",
    "VAR//Bilirubin_direct": "Direct bilirubin (mg/dL)",
    "VAR//Glucose": "Serum glucose (mg/dL)",
    "VAR//Lactate": "Lactic acid (mg/dL)",
    "VAR//Magnesium": "Magnesium (mmol/dL)",
    "VAR//Phosphate": "Phosphate (mg/dL)",
    "VAR//Potassium": "Potassium (mmol/L)",
    "VAR//Bilirubin_total": "Total bilirubin (mg/dL)",
    "VAR//TroponinI": "Troponin I (ng/mL)",
    "VAR//Hct": "Hematocrit (%)",
    "VAR//Hgb": "Hemoglobin (g/dL)",
    "VAR//PTT": "Partial thromboplastin time (seconds)",
    "VAR//WBC": "Leukocyte count (count * 10^3 / uL)",
    "VAR//Fibrinogen": "Fibrinogen (mg/dL)",
    "VAR//Platelets": "Platelet count (count * 10^3 / uL)",
    "AGE": "Age at ICU admission (years; 100 for patients over 89)",
    "SEX//M": "Male",
    "SEX//F": "Female",
    "UNIT//MICU": "Medical ICU (challenge column Unit1)",
    "UNIT//SICU": "Surgical ICU (challenge column Unit2)",
    "HOSP_ADM_TIME": "Hours between hospital admission and ICU admission (negative)",
    "ICU_HOUR": "Hour of the ICU stay, 1-based (challenge column ICULOS)",
    "MEDS_BIRTH": "Synthetic birth timestamp: hour 0 of the stay minus Age years",
    "SEPSIS_ONSET": "Sepsis-3 onset: first SepsisLabel == 1 hour plus 6 hours",
}

_STAY_RE = re.compile(r"p(\d+)\.psv$")

_SET_DIRS: tuple[str, ...] = ("training_setA", "training_setB")


def find_files(input_dir: str | Path) -> dict[str, list[Path]]:
    """Group the challenge PSVs under ``input_dir`` by training set.

    Both the released layout (``<root>/training/training_setA/p*.psv``) and a
    flattened one (``<root>/training_setA/p*.psv``) are accepted, as is a plain
    directory of PSVs, which is what the test fixtures are.
    """
    root = Path(input_dir)
    groups: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("p*.psv")):
        if _STAY_RE.search(path.name) is None:
            continue
        parent = path.parent.name
        groups.setdefault(parent if parent in _SET_DIRS else "training", []).append(path)
    if not groups:
        raise ValueError(f"no PhysioNet 2019 PSV files (p*.psv) found under {root}")
    return groups


def stay_id(path: Path) -> int:
    """``p000123.psv`` -> ``123``. The challenge ids are already disjoint per set."""
    match = _STAY_RE.search(path.name)
    if match is None:  # pragma: no cover - filtered by find_files
        raise ValueError(f"{path.name} is not a challenge-2019 stay file")
    return int(match.group(1))


def _scan(paths: Sequence[Path]) -> pl.LazyFrame:
    """Scan the PSVs with a pinned all-float schema and ``subject_id`` from the path."""
    return (
        pl.scan_csv(
            list(paths),
            separator="|",
            null_values=["NaN", "nan", ""],
            schema={name: pl.Float64 for name in PSV_COLUMNS},
            has_header=True,
            include_file_paths="_path",
        )
        .with_columns(
            subject_id=pl.col("_path").str.extract(r"p(\d+)\.psv$", 1).cast(pl.Int64, strict=False),
            time=(
                pl.lit(ORIGIN)
                + pl.duration(hours=(pl.col("ICULOS") - 1.0).cast(pl.Int64, strict=False))
            ).cast(pl.Datetime("us")),
        )
        .drop("_path")
        .drop_nulls(["subject_id", "time"])
    )


def _hourly_events(scan: pl.LazyFrame) -> pl.LazyFrame:
    """One event per non-null measurement, plus one ``ICU_HOUR`` marker per hour.

    ``ICULOS`` is unpivoted alongside the measurements rather than selected
    separately so the forty-thousand-file scan happens once instead of twice.
    The marker is kept because it is the only way an hour whose every
    measurement is missing shows up in the stream at all -- without it the hour
    grid has holes, and "at least six hours of history" stops being a fact the
    event stream can answer.
    """
    on = [*MEASUREMENTS, "ICULOS"]
    return (
        scan.select("subject_id", "time", *on)
        .unpivot(
            index=["subject_id", "time"],
            on=on,
            variable_name="_variable",
            value_name="numeric_value",
        )
        .drop_nulls("numeric_value")
        .select(
            "subject_id",
            "time",
            code=pl.when(pl.col("_variable") == "ICULOS")
            .then(pl.lit("ICU_HOUR"))
            .otherwise(pl.lit("VAR//") + pl.col("_variable")),
            numeric_value=pl.col("numeric_value").cast(pl.Float32),
        )
    )


def _per_stay(scan: pl.LazyFrame) -> pl.DataFrame:
    """One row per stay: statics, first flagged sepsis hour and hour count.

    Collected eagerly -- it is one row per stay, so at most a few tens of
    thousands -- because every static event and the onset event are built from
    it, and building them from separate lazy frames would mean re-reading every
    PSV once per event kind.
    """
    return (
        scan.group_by("subject_id")
        .agg(
            age=pl.col("Age").drop_nulls().first(),
            gender=pl.col("Gender").drop_nulls().first(),
            unit1=pl.col("Unit1").drop_nulls().max(),
            unit2=pl.col("Unit2").drop_nulls().max(),
            hosp_adm_time=pl.col("HospAdmTime").drop_nulls().first(),
            n_hours=pl.len(),
            first_flag=pl.col("time").filter(pl.col("SepsisLabel") == 1.0).min(),
        )
        .collect(engine="streaming")
        .sort("subject_id")
    )


def _static_tables(per_stay: pl.DataFrame) -> dict[str, pl.LazyFrame]:
    """Birth, age, sex, ICU unit, hospital-admission offset and sepsis onset."""
    frame = per_stay.lazy().with_columns(time=pl.lit(ORIGIN).cast(pl.Datetime("us")))
    tables: dict[str, pl.LazyFrame] = {
        "age": frame.drop_nulls("age").select(
            "subject_id",
            "time",
            code=pl.lit("AGE"),
            numeric_value=pl.col("age").cast(pl.Float32),
        ),
        "birth": frame.drop_nulls("age").select(
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
        "hosp_adm_time": frame.drop_nulls("hosp_adm_time").select(
            "subject_id",
            "time",
            code=pl.lit("HOSP_ADM_TIME"),
            numeric_value=pl.col("hosp_adm_time").cast(pl.Float32),
        ),
        "sepsis_onset": frame.drop_nulls("first_flag").select(
            "subject_id",
            time=(pl.col("first_flag") + pl.duration(hours=SEPSIS_SHIFT_HOURS)).cast(
                pl.Datetime("us")
            ),
            code=pl.lit("SEPSIS_ONSET"),
        ),
    }
    for column, code in (("unit1", "UNIT//MICU"), ("unit2", "UNIT//SICU")):
        tables[column] = frame.filter(pl.col(column) == 1.0).select(
            "subject_id", "time", code=pl.lit(code)
        )
    return tables


def _code_metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": list(DESCRIPTIONS),
            "description": list(DESCRIPTIONS.values()),
        },
        schema={"code": pl.String, "description": pl.String},
    )


def extract(input_dir: str | Path) -> SourceExtract:
    """Build a :class:`SourceExtract` for every challenge-2019 PSV under ``input_dir``."""
    input_dir = Path(input_dir)
    groups = find_files(input_dir)
    paths = [path for set_paths in groups.values() for path in set_paths]

    subject_ids = pl.Series("subject_id", sorted(stay_id(p) for p in paths), dtype=pl.Int64)
    if subject_ids.n_unique() != subject_ids.len():
        raise ValueError("duplicate stay ids across the challenge-2019 training sets")

    scan = _scan(paths)
    per_stay = _per_stay(scan)
    tables: dict[str, pl.LazyFrame] = {"hourly": _hourly_events(scan)}
    tables.update(_static_tables(per_stay))

    return SourceExtract(
        dataset_name="PhysioNet2019",
        source=str(input_dir),
        subject_ids=subject_ids,
        tables=tables,
        code_metadata=_code_metadata(),
        notes={
            **{f"{name}_stays": len(set_paths) for name, set_paths in groups.items()},
            "stays": len(paths),
            "septic_stays": int(per_stay["first_flag"].is_not_null().sum()),
            "recorded_hours": int(per_stay["n_hours"].sum()),
        },
    )
