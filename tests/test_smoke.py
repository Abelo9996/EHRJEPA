"""Smoke tests: the package imports, and we can write/read a valid MEDS shard."""

from __future__ import annotations

import datetime
import importlib

import meds
import polars as pl
import pyarrow.parquet as pq
import pytest

SUBPACKAGES = [
    "ehrjepa",
    "ehrjepa.data",
    "ehrjepa.models",
    "ehrjepa.objectives",
    "ehrjepa.train",
    "ehrjepa.eval",
    "ehrjepa.utils",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports_and_is_documented(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} is missing its module docstring"


def test_version_is_exposed() -> None:
    import ehrjepa

    assert isinstance(ehrjepa.__version__, str)
    assert ehrjepa.__version__


def _sample_events() -> pl.DataFrame:
    """A two-subject event stream in the shape the tokenizer will consume."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "time": [
                datetime.datetime(1994, 3, 2),
                datetime.datetime(2021, 7, 14, 9, 30),
                datetime.datetime(2021, 7, 14, 11, 5),
                datetime.datetime(1978, 11, 20),
                datetime.datetime(2022, 1, 3, 8, 0),
            ],
            "code": [
                meds.birth_code,
                "LAB//2160-0//mg/dL",
                "ICD10CM//I50.9",
                meds.birth_code,
                "LAB//718-7//g/dL",
            ],
            "numeric_value": [None, 1.1, None, None, 13.4],
        },
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float32,
        },
    )


def test_meds_data_shard_roundtrip(tmp_path) -> None:
    """A MEDS data shard survives a parquet round-trip and validates against the schema.

    This pins down the exact on-disk contract the data layer will target:
    ``subject_id`` int64, ``time`` timestamp[us], ``code`` string,
    ``numeric_value`` float32, sorted by (subject_id, time).
    """
    events = _sample_events().sort("subject_id", "time")
    shard = tmp_path / "data" / "train" / "0.parquet"
    shard.parent.mkdir(parents=True)
    events.write_parquet(shard)

    # polars emits arrow large_string for its String columns, so align to the
    # canonical MEDS arrow types before validating.
    table = meds.DataSchema.align(pq.read_table(shard))

    # The MEDS schema package is the authority on the layout; this raises on drift.
    meds.DataSchema.validate(table)

    assert table.schema.field("subject_id").type == meds.DataSchema.subject_id_dtype
    assert table.schema.field("code").type == meds.DataSchema.code_dtype
    assert table.schema.field("numeric_value").type == meds.DataSchema.numeric_value_dtype

    read_back = pl.read_parquet(shard)
    assert read_back.equals(events)
    assert read_back.height == 5
    assert read_back["subject_id"].n_unique() == 2

    # Per-subject streams are time-ordered, which the span sampler will rely on.
    for (_subject,), group in read_back.group_by("subject_id", maintain_order=True):
        assert group["time"].is_sorted()

    # numeric_value is genuinely nullable: coded events carry no value.
    assert read_back["numeric_value"].null_count() == 3
