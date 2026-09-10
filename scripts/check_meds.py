"""Check every parquet under MEDS extracts: ``python scripts/check_meds.py <meds_dir> [...]``."""

from __future__ import annotations

import glob
import sys

import polars as pl

bad = 0
for root in sys.argv[1:]:
    files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    rows = 0
    for f in files:
        try:
            rows += pl.scan_parquet(f).select(pl.len()).collect().item()
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"BAD {f}: {type(exc).__name__}")
    print(f"{root}: {len(files)} parquet files, {rows:,} rows")
sys.exit(1 if bad else 0)
