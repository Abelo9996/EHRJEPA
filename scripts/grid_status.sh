#!/usr/bin/env bash
# One-line status per grid: summary rows, finished flag, error count, and the
# last training step line.  Usage: scripts/grid_status.sh <run-name> [<run-name> ...]
cd "$(dirname "$0")/.."
for g in "$@"; do
  rows=$(grep -cE '^\| `' "docs/experiments/$g/summary.md" 2>/dev/null || echo 0)
  fin=$(grep -c 'ablation finished' "runs/$g/ablate.log" 2>/dev/null || echo 0)
  err=$(grep -cE 'Traceback|CUDA out of memory|Killed' "runs/$g/ablate.log" 2>/dev/null || echo 0)
  last=$(grep -E '^(step|\[)' "runs/$g/ablate.log" 2>/dev/null | tail -n 1 | cut -c1-70)
  echo "$g rows=$rows finished=$fin errors=$err | $last"
done
echo "runners=$(pgrep -c -f 'bin/python3? scripts/ablate.py' || true)"
