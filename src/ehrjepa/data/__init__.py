"""Data layer: source EHR -> MEDS -> event tokens -> batched tensors.

This subpackage will own everything between a raw clinical database and a tensor
the encoder can read. Planned contents: ETL wrappers that lower MIMIC-IV and
EHRSHOT extracts into the MEDS schema (``subject_id``, ``time``, ``code``,
``numeric_value``, plus dataset-specific extension columns) via ``meds_etl``,
validated against the ``meds`` schema package; a vocabulary builder that assigns
integer ids to codes with configurable frequency cutoffs and ontology rollup; a
tokenizer that turns a subject's event stream into a sequence of (code id,
numeric value, timestamp) triples with numeric values normalized per code; and
``torch.utils.data`` datasets and collators that sample context/target spans over
those sequences, pad to a common length, and emit the attention and loss masks
the JEPA objective needs. Parquet I/O goes through polars and pyarrow so that
subject-sharded MEDS data can be scanned lazily instead of loaded whole.
"""
