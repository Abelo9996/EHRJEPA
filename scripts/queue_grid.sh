#!/usr/bin/env bash
# Queue an ablation grid to start after another running grid finishes, inside tmux.
#
#   scripts/queue_grid.sh <wait-for-grid-yaml-substring> <grid.yaml> <run-name>
#
# Finds the PID of the running `scripts/ablate.py` whose command line contains
# the first argument, then launches <grid.yaml> via scripts/queue_after.sh in a
# detached tmux session named <run-name>, logging to runs/<run-name>/ablate.log.
set -euo pipefail
cd "$(dirname "$0")/.."
wait_for="$1"; grid="$2"; name="$3"
pid="$(pgrep -f "bin/python3? scripts/ablate.py.*${wait_for}" | head -n 1 || true)"
if [ -z "$pid" ]; then
  echo "no running ablate.py matching '${wait_for}'; refusing to launch (use scripts/launch_grid.sh)" >&2
  exit 1
fi
mkdir -p "runs/$name"
if tmux has-session -t "$name" 2>/dev/null; then
  echo "tmux session '$name' already exists"; exit 0
fi
tmux new-session -d -s "$name" \
  "scripts/queue_after.sh $pid uv run python scripts/ablate.py '$grid' > 'runs/$name/ablate.log' 2>&1"
sleep 2
tmux ls
echo "queued '$name' behind PID $pid; log: runs/$name/ablate.log"
