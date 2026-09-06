"""Drop one run's row from a grid summary so the runner re-evaluates it.

    python scripts/drop_row.py <grid-name> <run-name>

Removes the entry from ``docs/experiments/<grid>/summary.json`` and deletes
``docs/experiments/<grid>/eval/<run>/`` so that relaunching the grid re-scores
the cell.  Training artifacts under ``runs/`` are untouched; ``scripts/ablate.py``
reuses an existing ``final.pt`` rather than retraining.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    grid, run = sys.argv[1], sys.argv[2]
    doc_dir = REPO / "docs" / "experiments" / grid
    path = doc_dir / "summary.json"
    payload = json.loads(path.read_text())
    before = len(payload["runs"])
    payload["runs"] = [row for row in payload["runs"] if row.get("run") != run]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    eval_dir = doc_dir / "eval" / run
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    print(f"dropped {before - len(payload['runs'])} row(s) for {run!r} from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
