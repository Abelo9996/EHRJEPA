"""Source-specific ETLs that all land in one canonical MEDS layout.

Each submodule exposes ``extract(input_dir) -> SourceExtract``; the shared writer
in :mod:`ehrjepa.data.canonical` turns that into ``<out>/data/{train,tuning,
held_out}/*.parquet`` plus ``<out>/metadata/``. Splits are a deterministic 80/10/10
hash of ``subject_id``, so the same subject lands in the same split on every run
and on every machine, and no subject is ever split across shards.

Run one from the command line with::

    python -m ehrjepa.data.etl mimic --input <dir> --output <dir> [--shard-size N]

See ``src/ehrjepa/data/README.md`` for the layout and the per-source code
conventions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ehrjepa.data.canonical import SourceExtract

__all__ = ["SOURCES", "build_extract", "run"]

#: Source name -> callable building a :class:`SourceExtract` from an input directory.
SOURCES: dict[str, str] = {
    "mimic": "ehrjepa.data.etl.mimic",
    "desynpuf": "ehrjepa.data.etl.desynpuf",
    "synthea": "ehrjepa.data.etl.synthea",
}


def build_extract(source: str, input_dir: str | Path, **kwargs: object) -> SourceExtract:
    """Import the ETL for ``source`` and run its ``extract``."""
    from importlib import import_module

    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(SOURCES)}")
    module = import_module(SOURCES[source])
    extract: Callable[..., SourceExtract] = module.extract
    return extract(input_dir, **kwargs)


def run(
    source: str,
    input_dir: str | Path,
    output_dir: str | Path,
    shard_size: int | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """Extract ``source`` from ``input_dir`` and write canonical MEDS to ``output_dir``."""
    from ehrjepa.data.canonical import DEFAULT_SHARD_SIZE, write_canonical

    extract = build_extract(source, input_dir, **kwargs)
    return write_canonical(
        extract, output_dir, shard_size=shard_size if shard_size is not None else DEFAULT_SHARD_SIZE
    )
