"""CLI: ``python -m ehrjepa.data.etl <source> --input <dir> --output <dir>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ehrjepa.data.canonical import DEFAULT_SHARD_SIZE
from ehrjepa.data.etl import SOURCES, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ehrjepa.data.etl",
        description="Lower a raw EHR source into the canonical MEDS 0.4 layout.",
    )
    parser.add_argument("source", choices=sorted(SOURCES), help="which source ETL to run")
    parser.add_argument("--input", required=True, type=Path, help="source data directory")
    parser.add_argument("--output", required=True, type=Path, help="MEDS output directory")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help=f"subjects per output shard (default: {DEFAULT_SHARD_SIZE})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="scratch directory for the mimic source; reused if it already holds "
        "a finished meds_etl run (default: a temporary directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs: dict[str, object] = {}
    if args.source == "mimic" and args.work_dir is not None:
        kwargs["work_dir"] = args.work_dir

    started = time.monotonic()
    summary = run(args.source, args.input, args.output, shard_size=args.shard_size, **kwargs)
    summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
    summary["output"] = str(args.output)
    json.dump(summary, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
