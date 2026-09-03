"""Compatibility shim letting ``meds_etl`` 0.3.2 run against ``meds`` 0.4.

``meds`` 0.4 replaced the functional schema API (``meds.data_schema(...)``,
``meds.code_metadata_schema()``, ``meds.dataset_metadata_schema``) with the
``*Schema`` classes. ``meds_etl`` 0.3.2 -- the newest release that co-installs with
``meds`` 0.4 at all -- still calls the functional API, so importing and running it
unpatched dies with ``AttributeError: module 'meds' has no attribute 'data_schema'``
partway through the sort stage.

:func:`install` re-adds the three removed names, implemented on top of the 0.4
classes. It is idempotent and never overwrites an attribute that already exists,
so it becomes a no-op the moment ``meds_etl`` catches up to the class API.

Note that ``meds_etl`` 0.3.2 emits the pre-0.3.2 column name ``patient_id``; the
shim keeps that name (anything else would break the ``Table.cast`` it feeds) and
:mod:`ehrjepa.data.etl.mimic` renames it to ``subject_id`` on the way out.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import meds
import pyarrow as pa

__all__ = ["install"]

# meds_etl builds its shards with this column name; see meds_etl.unsorted.
_LEGACY_SUBJECT_COLUMN = "patient_id"


def _data_schema(custom_properties: Iterable[tuple[str, pa.DataType]] = ()) -> pa.Schema:
    """The MEDS 0.3-style data schema, with ``meds_etl``'s legacy id column name."""
    return pa.schema(
        [
            (_LEGACY_SUBJECT_COLUMN, pa.int64()),
            (meds.DataSchema.time_name, pa.timestamp("us")),
            (meds.DataSchema.code_name, pa.string()),
            (meds.DataSchema.numeric_value_name, pa.float32()),
            *list(custom_properties),
        ]
    )


def _code_metadata_schema(
    custom_per_code_properties: Sequence[tuple[str, pa.DataType]] = (),
) -> pa.Schema:
    return pa.schema([*list(meds.CodeMetadataSchema.schema()), *list(custom_per_code_properties)])


#: A permissive jsonschema standing in for the removed ``meds.dataset_metadata_schema``.
#: ``meds_etl`` only uses it via ``jsonschema.validate`` on a dict it built itself.
_DATASET_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        name: {"type": "string"} for name in meds.DatasetMetadataSchema.optional_columns()
    },
    "additionalProperties": True,
}


def install() -> None:
    """Add the removed ``meds`` 0.3 functional API back onto the ``meds`` module."""
    if not hasattr(meds, "data_schema"):
        meds.data_schema = _data_schema  # type: ignore[attr-defined]
    if not hasattr(meds, "code_metadata_schema"):
        meds.code_metadata_schema = _code_metadata_schema  # type: ignore[attr-defined]
    if not hasattr(meds, "dataset_metadata_schema"):
        meds.dataset_metadata_schema = _DATASET_METADATA_SCHEMA  # type: ignore[attr-defined]
