"""``python scripts/throughput.py`` -- steady-state tok/s for a few configurations.

Sizing an ablation grid needs one number per candidate: how many event tokens a
second of this laptop buys. Each cell here is a real 50-step training run through
:mod:`ehrjepa.train.pretrain` -- not a synthetic forward pass -- because the
masking, the SIGReg draw and the optimizer step are all part of what a step costs.

The first logging window is dropped from every cell: it contains lazy allocator
growth, the RoPE cache fill and the first Metal kernel compilations, and including
it understates a long run by a wide margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: ``(label, config, extra overrides)``.
CELLS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("small/jepa/fp32", "configs/pretrain_small.yaml", ("run.precision=fp32",)),
    ("small/jepa/bf16", "configs/pretrain_small.yaml", ("run.precision=bf16",)),
    ("pilot/jepa/fp32", "configs/pretrain_pilot.yaml", ("run.precision=fp32",)),
    ("pilot/jepa/bf16", "configs/pretrain_pilot.yaml", ("run.precision=bf16",)),
    ("pilot/ar/fp32", "configs/pretrain_pilot.yaml", ("run.precision=fp32", "objective.kind=ar")),
    ("pilot/ar/bf16", "configs/pretrain_pilot.yaml", ("run.precision=bf16", "objective.kind=ar")),
)


def measure(label: str, config: str, overrides: Sequence[str], steps: int, out_root: Path) -> dict:
    out_dir = out_root / label.replace("/", "-")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    command = [
        sys.executable,
        "-m",
        "ehrjepa.train.pretrain",
        "--config",
        config,
        "--override",
        f"run.steps={steps}",
        "run.log_every=10",
        "run.ckpt_every=0",
        "run.tensorboard=false",
        f"run.out_dir={out_dir}",
        *overrides,
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        return {"cell": label, "error": proc.stderr.strip().splitlines()[-1:] or ["failed"]}
    rows = list(csv.DictReader((out_dir / "metrics.csv").open()))
    steady = [float(r["tokens_per_s"]) for r in rows[1:]] or [float(rows[-1]["tokens_per_s"])]
    batch, max_len = _shape(out_dir)
    return {
        "cell": label,
        "config": config,
        "overrides": list(overrides),
        "batch_size": batch,
        "max_len": max_len,
        "steps": steps,
        "wall_s": round(wall, 1),
        "tokens_per_s": round(statistics.median(steady)),
        "tokens_per_s_min": round(min(steady)),
        "tokens_per_s_max": round(max(steady)),
        "peak_memory_mb": round(float(rows[-1]["peak_memory_mb"]), 1),
        "windows": len(steady),
    }


def _shape(out_dir: Path) -> tuple[int, int]:
    config = json.loads((out_dir / "config.json").read_text())
    return int(config["run"]["batch_size"]), int(config["data"]["max_len"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "throughput")
    parser.add_argument("--only", default=None, help="comma-separated cell labels")
    args = parser.parse_args(list(argv) if argv is not None else None)

    wanted = set(args.only.split(",")) if args.only else None
    results = []
    for label, config, overrides in CELLS:
        if wanted and label not in wanted:
            continue
        row = measure(label, config, overrides, args.steps, args.out)
        results.append(row)
        print(json.dumps(row), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "throughput.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
