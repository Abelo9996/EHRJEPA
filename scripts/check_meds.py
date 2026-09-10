"""Print shard shapes for MEDS extracts: ``python scripts/check_meds.py <meds_dir> [...]``."""

from __future__ import annotations

import glob
import sys

import polars as pl

for root in sys.argv[1:]:
    files = sorted(glob.glob(f"{root}/data/*/*.parquet"))
    rows = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    print(f"{root}: {len(files)} shards, {rows:,} rows")
