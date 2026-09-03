"""End-to-end tests for the three source ETLs and the canonical MEDS writer.

Every test runs against the hand-written fixtures in ``tests/fixtures/`` -- a few
rows per source, chosen to hit the awkward cases (a null discharge date, a
condition with no stop, a text-valued observation, a beneficiary who only appears
in the drug-event file). The whole module is expected to run in a couple of
seconds; nothing here touches ``data/``.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

import meds
import polars as pl
import pyarrow.parquet as pq
import pytest

from ehrjepa.data import etl, stats
from ehrjepa.data.canonical import (
    SPLITS,
    assign_subjects,
    coerce_events,
    shard_for,
    split_for,
    stable_subject_id,
    write_canonical,
)
from ehrjepa.data.etl import desynpuf, mimic

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# Determinism of the split / shard / subject-id hashes
# --------------------------------------------------------------------------- #


def test_split_assignment_is_deterministic_and_stable() -> None:
    """The split hash is pinned: these values must not move between releases."""
    first = [split_for(i) for i in range(50)]
    second = [split_for(i) for i in range(50)]
    assert first == second
    # A literal regression pin. If this changes, every existing extract reshuffles.
    assert split_for(0) == "train"
    assert [split_for(i) for i in (7, 13, 31)] == ["tuning", "held_out", "held_out"]


def test_split_ratios_are_close_to_80_10_10() -> None:
    counts = Counter(split_for(i) for i in range(20_000))
    assert set(counts) == set(SPLITS)
    assert 0.78 < counts["train"] / 20_000 < 0.82
    assert 0.09 < counts["tuning"] / 20_000 < 0.11
    assert 0.09 < counts["held_out"] / 20_000 < 0.11


def test_stable_subject_id_is_deterministic_and_positive() -> None:
    assert stable_subject_id("AAAA000000000001") == stable_subject_id("AAAA000000000001")
    assert stable_subject_id("AAAA000000000001") != stable_subject_id("AAAA000000000002")
    for source_id in ("a", "AAAA000000000001", "11111111-1111-1111-1111-111111111111"):
        value = stable_subject_id(source_id)
        assert 0 <= value < 2**63


def test_shard_assignment_is_patient_disjoint_and_bounded() -> None:
    ids = pl.Series("subject_id", range(1000), dtype=pl.Int64)
    assignment = assign_subjects(ids, shard_size=100)
    assert assignment.height == 1000
    assert assignment["subject_id"].n_unique() == 1000
    # One (split, shard) per subject: shards never straddle splits.
    per_split = assignment.group_by("split").agg(pl.col("shard").max().alias("max_shard"))
    for row in per_split.to_dicts():
        n_subjects = assignment.filter(pl.col("split") == row["split"]).height
        assert row["max_shard"] < max(1, -(-n_subjects // 100))
    assert all(shard_for(i, 1) == 0 for i in range(20))


def test_coerce_events_fills_optionals_and_drops_unusable_rows() -> None:
    frame = pl.LazyFrame(
        {
            "subject_id": [1, 2, None, 4],
            "time": [dt.datetime(2020, 1, 1)] * 4,
            "code": ["A", None, "C", ""],
            "extra": [1, 2, 3, 4],
        }
    )
    out = coerce_events(frame).collect()
    assert out.columns == ["subject_id", "time", "code", "numeric_value", "text_value"]
    assert out["subject_id"].to_list() == [1]
    assert out.schema["numeric_value"] == pl.Float32


def test_coerce_events_rejects_a_frame_without_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        coerce_events(pl.LazyFrame({"subject_id": [1]})).collect()


# --------------------------------------------------------------------------- #
# Shared assertions about a canonical MEDS directory
# --------------------------------------------------------------------------- #


def assert_canonical(out: Path) -> pl.DataFrame:
    """Validate the layout and return every event, concatenated."""
    assert (out / meds.dataset_metadata_filepath).exists()
    assert (out / meds.code_metadata_filepath).exists()
    assert (out / meds.subject_splits_filepath).exists()

    metadata = json.loads((out / meds.dataset_metadata_filepath).read_text())
    assert metadata["meds_version"] == meds.__version__
    assert metadata["etl_name"] == "ehrjepa.data.etl"
    assert {"dataset_name", "etl_version", "source", "created"} <= set(metadata)

    splits = pl.read_parquet(out / meds.subject_splits_filepath)
    meds.SubjectSplitSchema.validate(meds.SubjectSplitSchema.align(splits.to_arrow()))
    assert set(splits["split"]) <= set(SPLITS)
    # Patient-disjoint splits: one row per subject.
    assert splits["subject_id"].n_unique() == splits.height

    meds.CodeMetadataSchema.validate(
        meds.CodeMetadataSchema.align(pq.read_table(out / meds.code_metadata_filepath))
    )

    frames = []
    for split in SPLITS:
        assert (out / "data" / split).is_dir(), f"missing split directory {split}"
        for shard in sorted((out / "data" / split).glob("*.parquet")):
            table = pq.read_table(shard)
            # Every shard is schema-valid on its own.
            meds.DataSchema.validate(meds.DataSchema.align(table))
            frame = pl.from_arrow(table)
            assert frame.columns[:3] == ["subject_id", "time", "code"]

            # Sort order: subject_id then time, with null (static) times first.
            keys = frame.select("subject_id", "time")
            assert keys.equals(keys.sort("subject_id", "time", nulls_last=False))

            # Shards are subject-disjoint and split-consistent.
            shard_subjects = set(frame["subject_id"].unique())
            assigned = set(splits.filter(pl.col("split") == split)["subject_id"])
            assert shard_subjects <= assigned
            frames.append(frame.with_columns(pl.lit(split).alias("split")))

    events = pl.concat(frames) if frames else pl.DataFrame()

    # The vocabulary in metadata is exactly the vocabulary in the data.
    codes = pl.read_parquet(out / meds.code_metadata_filepath)
    assert set(codes["code"]) == set(events["code"].unique())

    # No subject appears in two splits.
    per_subject = events.group_by("subject_id").agg(pl.col("split").n_unique().alias("n"))
    assert per_subject["n"].max() == 1
    return events


# --------------------------------------------------------------------------- #
# Synthea
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthea_out(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthea") / "meds"
    etl.run("synthea", FIXTURES / "synthea", out, shard_size=2)
    return out


def test_synthea_end_to_end(synthea_out: Path) -> None:
    events = assert_canonical(synthea_out)
    assert events["subject_id"].n_unique() == 3

    by_code = dict(events.group_by("code").len().iter_rows())
    assert by_code["MEDS_BIRTH"] == 3
    assert by_code["MEDS_DEATH"] == 1
    assert by_code["SEX//M"] == 1
    assert by_code["RACE//WHITE"] == 1
    assert by_code["ETHNICITY//NONHISPANIC"] == 2
    assert by_code["ENCOUNTER//WELLNESS"] == 1
    assert by_code["ENCOUNTER//INPATIENT"] == 1
    assert by_code["END//ENCOUNTER"] == 3
    assert by_code["SNOMED//15777000"] == 1
    assert by_code["CONDITION_STOP//15777000"] == 1
    assert "CONDITION_STOP//271737000" not in by_code  # that condition has no stop date
    assert by_code["SNOMED_PROC//73761001"] == 1
    assert by_code["RXNORM//310965"] == 1


def test_synthea_units_are_folded_into_the_observation_code(synthea_out: Path) -> None:
    events = pl.concat([pl.read_parquet(p) for p in sorted(synthea_out.rglob("data/*/*.parquet"))])
    height = events.filter(pl.col("code") == "LOINC//8302-2//cm")
    assert height.height == 1
    assert height["numeric_value"].item() == pytest.approx(182.1, rel=1e-5)

    # A non-numeric observation keeps its text and gets the UNK unit slot.
    smoking = events.filter(pl.col("code") == "LOINC//72166-2//UNK")
    assert smoking.height == 1
    assert smoking["numeric_value"].item() is None
    assert smoking["text_value"].item() == "Never smoker"


def test_synthea_medication_carries_dispenses(synthea_out: Path) -> None:
    events = pl.concat([pl.read_parquet(p) for p in sorted(synthea_out.rglob("data/*/*.parquet"))])
    row = events.filter(pl.col("code") == "RXNORM//314076")
    assert row["numeric_value"].item() == pytest.approx(12.0)


def test_synthea_id_map_round_trips(synthea_out: Path) -> None:
    id_map = pl.read_parquet(synthea_out / "metadata" / "subject_id_map.parquet")
    assert id_map.height == 3
    for source_id, subject_id in id_map.iter_rows():
        assert stable_subject_id(source_id) == subject_id


# --------------------------------------------------------------------------- #
# DE-SynPUF
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def desynpuf_out(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("desynpuf") / "meds"
    etl.run("desynpuf", FIXTURES / "desynpuf", out, shard_size=2)
    return out


def test_desynpuf_end_to_end(desynpuf_out: Path) -> None:
    events = assert_canonical(desynpuf_out)
    # DDDD only appears in the PDE file; it must still become a subject.
    assert events["subject_id"].n_unique() == 4

    by_code = dict(events.group_by("code").len().iter_rows())
    assert by_code["MEDS_BIRTH"] == 3
    assert by_code["MEDS_DEATH"] == 1
    assert by_code["SEX//M"] == 2
    assert by_code["SEX//F"] == 1
    assert by_code["RACE//HISPANIC"] == 1
    assert by_code["ADMISSION//INPATIENT"] == 2
    assert by_code["DISCHARGE//INPATIENT"] == 2
    assert by_code["VISIT//OUTPATIENT"] == 2
    assert by_code["DRG//470"] == 1
    assert by_code["ICD9CM//4019"] == 2
    assert by_code["ICD9PROC//3961"] == 1
    assert by_code["HCPCS//85610"] == 1
    assert by_code["NDC//00247037252"] == 2


def test_desynpuf_chronic_flags_only_fire_when_set(desynpuf_out: Path) -> None:
    events = pl.concat([pl.read_parquet(p) for p in sorted(desynpuf_out.rglob("data/*/*.parquet"))])
    flags = events.filter(pl.col("code").str.starts_with("SP_"))
    # AAAA: alzheimer's in both years, CHF only in 2009. BBBB: diabetes in both.
    assert dict(flags.group_by("code").len().iter_rows()) == {
        "SP_ALZHDMTA//1": 2,
        "SP_CHF//1": 1,
        "SP_DIABETES//1": 2,
    }
    # Flags are dated 1 January of the summary year.
    assert set(flags["time"].dt.month().unique()) == {1}
    assert set(flags["time"].dt.day().unique()) == {1}
    assert set(flags["time"].dt.year().unique()) == {2008, 2009}


def test_desynpuf_length_of_stay_and_admission_dates(desynpuf_out: Path) -> None:
    events = pl.concat([pl.read_parquet(p) for p in sorted(desynpuf_out.rglob("data/*/*.parquet"))])
    discharge = events.filter(pl.col("code") == "DISCHARGE//INPATIENT").sort("time")
    assert discharge["numeric_value"].to_list() == pytest.approx([5.0, 6.0])
    admission = events.filter(pl.col("code") == "ADMISSION//INPATIENT").sort("time")
    assert admission["time"].to_list() == [dt.datetime(2008, 3, 10), dt.datetime(2009, 4, 12)]


def test_desynpuf_pde_carries_days_supplied(desynpuf_out: Path) -> None:
    events = pl.concat([pl.read_parquet(p) for p in sorted(desynpuf_out.rglob("data/*/*.parquet"))])
    fills = events.filter(pl.col("code").str.starts_with("NDC//")).sort("time")
    assert fills["numeric_value"].to_list() == pytest.approx([20.0, 30.0, 90.0, 30.0])


def test_desynpuf_file_discovery() -> None:
    groups = desynpuf.find_files(FIXTURES / "desynpuf")
    assert [len(v) for v in (groups["beneficiary"], groups["inpatient"])] == [2, 1]
    assert len(groups["outpatient"]) == 1 and len(groups["pde"]) == 1


def test_desynpuf_extract_tables_are_lazy() -> None:
    """The extract must hand the writer lazy frames, or it will not scale."""
    extract = desynpuf.extract(FIXTURES / "desynpuf")
    assert extract.tables
    assert all(isinstance(frame, pl.LazyFrame) for frame in extract.tables.values())


# --------------------------------------------------------------------------- #
# MIMIC-IV (the repartitioning half; the meds_etl half needs the real release)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def mimic_out(tmp_path_factory) -> Path:
    """Build a stand-in ``meds_etl`` output directory and repartition it.

    Running ``meds_etl.mimic`` itself needs all 23 MIMIC-IV tables, so the unit
    test starts from a fixture shaped like its output -- legacy ``patient_id``
    column, extra per-source metadata columns and all.
    """
    root = tmp_path_factory.mktemp("mimic")
    meds_etl_dir = root / "meds_etl_out"
    (meds_etl_dir / "data").mkdir(parents=True)
    (meds_etl_dir / "metadata").mkdir(parents=True)

    frame = pl.read_csv(FIXTURES / "mimic" / "meds_etl_data.csv", infer_schema_length=0)
    frame.with_columns(
        pl.col("patient_id").cast(pl.Int64),
        pl.col("time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S"),
        pl.col("numeric_value").cast(pl.Float32, strict=False),
        pl.col("hadm_id").cast(pl.Int64, strict=False),
    ).write_parquet(meds_etl_dir / "data" / "data_0.parquet")
    pl.DataFrame(
        {
            "code": ["ICD9CM/25000"],
            "description": ["Diabetes mellitus without mention of complication"],
            "parent_codes": [["ICD9CM/250.00"]],
        }
    ).write_parquet(meds_etl_dir / "metadata" / "codes.parquet")

    extract = mimic.from_meds_etl_output(meds_etl_dir, source="fixture")
    out = root / "meds"
    write_canonical(extract, out, shard_size=1)
    return out


def test_mimic_repartition_end_to_end(mimic_out: Path) -> None:
    events = assert_canonical(mimic_out)
    assert events["subject_id"].n_unique() == 2
    assert set(events["subject_id"]) == {10000032, 10000084}
    # Extra meds_etl metadata columns are dropped in favour of one column set.
    assert "hadm_id" not in events.columns
    assert "unit" not in events.columns
    # text_value survives.
    assert events.filter(pl.col("code") == "MIMIC_IV_LAB/50940")["text_value"].item() == "NEG"
    assert events.filter(pl.col("code") == "LOINC/1975-2")["numeric_value"].item() == pytest.approx(
        0.4
    )


def test_mimic_code_metadata_descriptions_are_carried_over(mimic_out: Path) -> None:
    codes = pl.read_parquet(mimic_out / meds.code_metadata_filepath)
    described = codes.filter(pl.col("code") == "ICD9CM/25000")
    assert described["description"].item().startswith("Diabetes mellitus")
    assert described["parent_codes"].item().to_list() == ["ICD9CM/250.00"]
    # Codes with no supplied metadata still appear, with a null description.
    assert codes.filter(pl.col("code") == "MEDS_BIRTH")["description"].item() is None


def test_meds_compat_shim_is_idempotent() -> None:
    import pyarrow as pa

    from ehrjepa.data.etl import _meds_compat

    _meds_compat.install()
    _meds_compat.install()
    assert callable(meds.data_schema)
    schema = meds.data_schema([("hadm_id", pa.int64())])
    assert schema.names == ["patient_id", "time", "code", "numeric_value", "hadm_id"]
    assert meds.code_metadata_schema().names == ["code", "description", "parent_codes"]
    assert meds.dataset_metadata_schema["type"] == "object"


# --------------------------------------------------------------------------- #
# Cross-source and CLI
# --------------------------------------------------------------------------- #


def test_split_assignment_is_reproducible_across_runs(tmp_path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    etl.run("synthea", FIXTURES / "synthea", first, shard_size=2)
    etl.run("synthea", FIXTURES / "synthea", second, shard_size=2)
    assert pl.read_parquet(first / meds.subject_splits_filepath).equals(
        pl.read_parquet(second / meds.subject_splits_filepath)
    )
    for split in SPLITS:
        for shard in sorted((first / "data" / split).glob("*.parquet")):
            other = second / "data" / split / shard.name
            assert pl.read_parquet(shard).equals(pl.read_parquet(other))


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        etl.build_extract("ehrshot", ".")


def test_etl_cli(tmp_path, capsys) -> None:
    from ehrjepa.data.etl.__main__ import main

    out = tmp_path / "meds"
    assert (
        main(
            [
                "synthea",
                "--input",
                str(FIXTURES / "synthea"),
                "--output",
                str(out),
                "--shard-size",
                "2",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["subjects"] == 3
    assert summary["events"] > 0
    assert (out / meds.dataset_metadata_filepath).exists()


def test_stats_cli_and_computation(synthea_out: Path, capsys) -> None:
    computed = stats.compute_stats(synthea_out, top_k=5)
    assert computed["subjects"] == 3
    assert computed["vocab_size"] > 0
    assert len(computed["top_codes"]) == 5
    assert set(computed["per_split"]) <= set(SPLITS)
    assert computed["checks"]["events_before_birth"] == 0
    assert computed["checks"]["subjects_without_birth"] == 0

    assert stats.main([str(synthea_out), "--top-k", "3"]) == 0
    rendered = capsys.readouterr().out
    assert "### Summary" in rendered and "### Integrity checks" in rendered

    assert stats.main([str(synthea_out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["subjects"] == 3
