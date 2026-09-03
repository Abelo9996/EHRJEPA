"""Data layer: source EHR -> MEDS -> event tokens -> batched tensors.

This subpackage owns everything between a raw clinical database and a tensor the
encoder can read, in four stages, each of which is the input contract of the
next. ``src/ehrjepa/data/README.md`` is the authority on the on-disk shapes.

* :mod:`~ehrjepa.data.etl` lowers MIMIC-IV, CMS DE-SynPUF and Synthea into the
  MEDS 0.4 schema (``subject_id``, ``time``, ``code``, ``numeric_value``,
  ``text_value``), and :mod:`~ehrjepa.data.canonical` writes them all into one
  identical layout with deterministic, subject-disjoint splits and shards.
* :mod:`~ehrjepa.data.stats` reports what came out, including the integrity
  checks (events before birth, after death, null times) every real extract needs.
* :mod:`~ehrjepa.data.tokenize` fits, on the train split only, the vocabulary
  (with hierarchical rollup of rare codes before ``UNK``) and the per-code value
  quantizer, and :mod:`~ehrjepa.data.cache` uses them to lower every split into
  flat memmap-friendly arrays: one token per event, carrying code, value and
  time.
* :mod:`~ehrjepa.data.dataset` samples context windows over that cache and
  collates them into padded batches with attention masks.

Parquet I/O goes through polars and pyarrow so subject-sharded MEDS data can be
scanned lazily instead of loaded whole.
"""
