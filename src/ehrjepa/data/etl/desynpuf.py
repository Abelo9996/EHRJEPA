"""CMS DE-SynPUF -> canonical MEDS.

DE-SynPUF ships four kinds of file per sample: a beneficiary summary per calendar
year, inpatient claims, outpatient claims and prescription drug events (PDE).
Everything is a quoted CSV with ``YYYYMMDD`` dates and no types, and the claim
tables are extremely wide -- an inpatient claim carries 10 diagnosis, 6 procedure
and 45 HCPCS columns, almost all null.

The whole module is lazy: files are ``scan_csv``-ed with schema inference off (so
every column is a string and the wide code blocks share a dtype for ``unpivot``),
the wide blocks are melted long, nulls are dropped, and the resulting frames are
handed to the canonical writer's streaming sink. Nothing but the beneficiary id
inventory is ever materialised, so all 20 samples fit the same code path.

Code conventions
----------------
``MEDS_BIRTH`` / ``MEDS_DEATH``; ``SEX//{M,F}``; ``RACE//{WHITE,BLACK,OTHER,HISPANIC}``;
``SP_<FLAG>//1`` dated 1 January of the summary year; ``ADMISSION//INPATIENT`` /
``DISCHARGE//INPATIENT`` (the latter carrying length of stay in days as its
numeric value); ``VISIT//OUTPATIENT``; ``DRG//<code>``; ``ICD9CM//<code>``;
``ICD9PROC//<code>``; ``HCPCS//<code>``; ``NDC//<code>`` carrying days supplied.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl

from ehrjepa.data.canonical import SourceExtract, stable_subject_id

__all__ = ["extract", "find_files"]

_ID = "DESYNPUF_ID"

#: Chronic-condition flag columns on the beneficiary summary. 1 = yes, 2 = no.
CHRONIC_FLAGS: tuple[str, ...] = (
    "SP_ALZHDMTA",
    "SP_CHF",
    "SP_CHRNKIDN",
    "SP_CNCR",
    "SP_COPD",
    "SP_DEPRESSN",
    "SP_DIABETES",
    "SP_ISCHMCHT",
    "SP_OSTEOPRS",
    "SP_RA_OA",
    "SP_STRKETIA",
)

#: CMS code books. Unmapped values fall through as the raw code.
SEX_LABELS = {"1": "M", "2": "F"}
RACE_LABELS = {"1": "WHITE", "2": "BLACK", "3": "OTHER", "5": "HISPANIC"}

_DX_COLUMNS = tuple(f"ICD9_DGNS_CD_{i}" for i in range(1, 11))
_PROC_COLUMNS = tuple(f"ICD9_PRCDR_CD_{i}" for i in range(1, 7))
_HCPCS_COLUMNS = tuple(f"HCPCS_CD_{i}" for i in range(1, 46))

_YEAR_RE = re.compile(r"_((?:19|20)\d{2})_")


def _scan(path: Path) -> pl.LazyFrame:
    """Scan a DE-SynPUF CSV with every column typed as a string."""
    return pl.scan_csv(path, infer_schema_length=0, quote_char='"')


def _date(column: str) -> pl.Expr:
    return pl.col(column).str.strptime(pl.Datetime("us"), "%Y%m%d", strict=False)


def _clean(expr: pl.Expr) -> pl.Expr:
    """Trim a raw code and turn the empty string into a null."""
    trimmed = expr.str.strip_chars()
    return pl.when(trimmed.str.len_chars() > 0).then(trimmed).otherwise(None)


def find_files(input_dir: str | Path) -> dict[str, list[Path]]:
    """Group the DE-SynPUF CSVs under ``input_dir`` (recursively) by kind."""
    root = Path(input_dir)
    groups: dict[str, list[Path]] = {
        "beneficiary": [],
        "inpatient": [],
        "outpatient": [],
        "pde": [],
    }
    for path in sorted(root.rglob("*.csv")):
        name = path.name.lower()
        if "beneficiary_summary" in name:
            groups["beneficiary"].append(path)
        elif "inpatient_claims" in name:
            groups["inpatient"].append(path)
        elif "outpatient_claims" in name:
            groups["outpatient"].append(path)
        elif "prescription_drug_events" in name:
            groups["pde"].append(path)
    if not any(groups.values()):
        raise ValueError(f"no DE-SynPUF CSVs found under {root}")
    return groups


def _summary_year(path: Path) -> int:
    match = _YEAR_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot read a summary year out of {path.name}")
    return int(match.group(1))


def _id_map(paths: Iterable[Path]) -> pl.DataFrame:
    """Build the ``DESYNPUF_ID`` -> Int64 ``subject_id`` mapping over every file.

    Built from *all* files, not just the beneficiary summaries: a claim may name a
    beneficiary absent from the summary years present on disk, and dropping those
    silently would lose events.
    """
    ids = (
        pl.concat([_scan(p).select(pl.col(_ID)) for p in paths], how="vertical")
        .unique()
        .collect(engine="streaming")
        .to_series()
        .drop_nulls()
        .sort()
    )
    mapped = pl.DataFrame(
        {
            "source_id": ids,
            "subject_id": pl.Series(
                [stable_subject_id(str(value)) for value in ids], dtype=pl.Int64
            ),
        }
    )
    if mapped["subject_id"].n_unique() != mapped.height:
        raise RuntimeError("subject id hash collision in DE-SynPUF id map")
    return mapped


def _with_subject(frame: pl.LazyFrame, id_map: pl.LazyFrame) -> pl.LazyFrame:
    return frame.join(id_map, left_on=_ID, right_on="source_id", how="inner")


def _melt_codes(
    frame: pl.LazyFrame, columns: Sequence[str], prefix: str, extra: Sequence[str] = ()
) -> pl.LazyFrame:
    """Melt a wide block of code columns into one event row per non-null code."""
    index = ["subject_id", "time", *extra]
    present = [c for c in columns if c in frame.collect_schema().names()]
    if not present:
        return pl.LazyFrame(
            schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.String}
        )
    return (
        frame.select([*index, *present])
        .unpivot(index=index, on=present, variable_name="_column", value_name="_code")
        .with_columns(_clean(pl.col("_code")).alias("_code"))
        .drop_nulls("_code")
        .select(
            pl.col("subject_id"),
            pl.col("time"),
            (pl.lit(f"{prefix}//") + pl.col("_code")).alias("code"),
        )
    )


def _demographic_events(paths: Sequence[Path], id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    """Birth, death, sex, race and per-year chronic-condition flag events."""
    tables: dict[str, pl.LazyFrame] = {}

    per_year = [
        _with_subject(_scan(path), id_map).with_columns(pl.lit(_summary_year(path)).alias("_year"))
        for path in paths
    ]
    summaries = pl.concat(per_year, how="vertical")

    # A beneficiary appears once per summary year. Reduce those rows to one:
    # earliest birth, latest death, earliest year's sex/race. The death date is
    # *not* backfilled into earlier summary years -- it is only populated in the
    # year the beneficiary died -- so a keep-first dedupe would silently drop
    # every death in the dataset.
    demographics = (
        summaries.select(
            "subject_id",
            "_year",
            _date("BENE_BIRTH_DT").alias("_birth"),
            _date("BENE_DEATH_DT").alias("_death"),
            _clean(pl.col("BENE_SEX_IDENT_CD")).alias("_sex"),
            _clean(pl.col("BENE_RACE_CD")).alias("_race"),
        )
        .sort("_year")
        .group_by("subject_id", maintain_order=True)
        .agg(
            pl.col("_birth").min(),
            pl.col("_death").max(),
            pl.col("_sex").drop_nulls().first(),
            pl.col("_race").drop_nulls().first(),
        )
    )

    tables["birth"] = demographics.select(
        "subject_id", pl.col("_birth").alias("time"), pl.lit("MEDS_BIRTH").alias("code")
    ).drop_nulls("time")
    tables["death"] = demographics.select(
        "subject_id", pl.col("_death").alias("time"), pl.lit("MEDS_DEATH").alias("code")
    ).drop_nulls("time")
    tables["sex"] = demographics.drop_nulls("_sex").select(
        "subject_id",
        pl.col("_birth").alias("time"),
        (pl.lit("SEX//") + pl.col("_sex").replace(SEX_LABELS)).alias("code"),
    )
    tables["race"] = demographics.drop_nulls("_race").select(
        "subject_id",
        pl.col("_birth").alias("time"),
        (pl.lit("RACE//") + pl.col("_race").replace(RACE_LABELS)).alias("code"),
    )

    present_flags = [f for f in CHRONIC_FLAGS if f in summaries.collect_schema().names()]
    if present_flags:
        flags = summaries.select(
            "subject_id",
            pl.datetime(pl.col("_year"), 1, 1).cast(pl.Datetime("us")).alias("time"),
            *[pl.col(flag) for flag in present_flags],
        )
        tables["chronic_flags"] = (
            flags.unpivot(
                index=["subject_id", "time"],
                on=present_flags,
                variable_name="_flag",
                value_name="_value",
            )
            .filter(pl.col("_value").str.strip_chars() == "1")
            .select("subject_id", "time", (pl.col("_flag") + pl.lit("//1")).alias("code"))
        )
    return tables


def _inpatient_events(paths: Sequence[Path], id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    claims = _with_subject(
        pl.concat([_scan(p) for p in paths], how="vertical"), id_map
    ).with_columns(
        pl.coalesce(_date("CLM_ADMSN_DT"), _date("CLM_FROM_DT")).alias("time"),
        _date("NCH_BENE_DSCHRG_DT").alias("_discharge"),
    )

    tables: dict[str, pl.LazyFrame] = {
        "ip_admission": claims.drop_nulls("time").select(
            "subject_id", "time", pl.lit("ADMISSION//INPATIENT").alias("code")
        ),
        "ip_discharge": claims.drop_nulls("_discharge").select(
            "subject_id",
            pl.col("_discharge").alias("time"),
            pl.lit("DISCHARGE//INPATIENT").alias("code"),
            (pl.col("_discharge") - pl.col("time"))
            .dt.total_days()
            .cast(pl.Float32)
            .alias("numeric_value"),
        ),
        "ip_drg": claims.drop_nulls("time")
        .with_columns(_clean(pl.col("CLM_DRG_CD")).alias("_drg"))
        .drop_nulls("_drg")
        .select("subject_id", "time", (pl.lit("DRG//") + pl.col("_drg")).alias("code")),
    }
    dated = claims.drop_nulls("time")
    tables["ip_dx"] = _melt_codes(dated, ("ADMTNG_ICD9_DGNS_CD", *_DX_COLUMNS), "ICD9CM")
    tables["ip_proc"] = _melt_codes(dated, _PROC_COLUMNS, "ICD9PROC")
    tables["ip_hcpcs"] = _melt_codes(dated, _HCPCS_COLUMNS, "HCPCS")
    return tables


def _outpatient_events(paths: Sequence[Path], id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    claims = (
        _with_subject(pl.concat([_scan(p) for p in paths], how="vertical"), id_map)
        .with_columns(_date("CLM_FROM_DT").alias("time"))
        .drop_nulls("time")
    )
    return {
        "op_visit": claims.select("subject_id", "time", pl.lit("VISIT//OUTPATIENT").alias("code")),
        "op_dx": _melt_codes(claims, ("ADMTNG_ICD9_DGNS_CD", *_DX_COLUMNS), "ICD9CM"),
        "op_proc": _melt_codes(claims, _PROC_COLUMNS, "ICD9PROC"),
        "op_hcpcs": _melt_codes(claims, _HCPCS_COLUMNS, "HCPCS"),
    }


def _pde_events(paths: Sequence[Path], id_map: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    """One ``NDC//<code>`` event per fill, valued with days supplied.

    ``QTY_DSPNSD_NUM`` is deliberately *not* emitted as a second event: quantity is
    on a per-product scale (tablets vs. mL vs. grams) that is not comparable across
    NDCs without a package-size join, whereas days supplied is comparable and is
    what downstream exposure models want. Quantity remains in the source file.
    """
    fills = (
        _with_subject(pl.concat([_scan(p) for p in paths], how="vertical"), id_map)
        .with_columns(
            _date("SRVC_DT").alias("time"),
            _clean(pl.col("PROD_SRVC_ID")).alias("_ndc"),
        )
        .drop_nulls(["time", "_ndc"])
    )
    return {
        "pde": fills.select(
            "subject_id",
            "time",
            (pl.lit("NDC//") + pl.col("_ndc")).alias("code"),
            pl.col("DAYS_SUPLY_NUM").cast(pl.Float32, strict=False).alias("numeric_value"),
        )
    }


def extract(input_dir: str | Path) -> SourceExtract:
    """Build a :class:`SourceExtract` for every DE-SynPUF sample under ``input_dir``."""
    input_dir = Path(input_dir)
    groups = find_files(input_dir)
    all_paths = [p for paths in groups.values() for p in paths]

    id_map = _id_map(all_paths)
    id_map_lazy = id_map.lazy()

    tables: dict[str, pl.LazyFrame] = {}
    if groups["beneficiary"]:
        tables.update(_demographic_events(groups["beneficiary"], id_map_lazy))
    if groups["inpatient"]:
        tables.update(_inpatient_events(groups["inpatient"], id_map_lazy))
    if groups["outpatient"]:
        tables.update(_outpatient_events(groups["outpatient"], id_map_lazy))
    if groups["pde"]:
        tables.update(_pde_events(groups["pde"], id_map_lazy))

    return SourceExtract(
        dataset_name="CMS DE-SynPUF",
        source=str(input_dir),
        subject_ids=id_map["subject_id"],
        tables=tables,
        id_map=id_map,
        notes={f"{kind}_files": len(paths) for kind, paths in groups.items()},
    )
