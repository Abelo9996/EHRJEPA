#!/usr/bin/env bash
# Launch an ablation grid detached inside tmux so it survives SSH disconnects.
#
#   scripts/launch_grid.sh <grid.yaml> <run-name> [extra ablate.py args...]
#
# Logs to runs/<run-name>/ablate.log; the tmux session is named <run-name>.
# Re-running the same command after a disconnect resumes the grid (the runner
# skips cells that already have a summary row).
set -euo pipefail
cd "$(dirname "$0")/.."
grid="$1"; name="$2"; shift 2
mkdir -p "runs/$name"
if tmux has-session -t "$name" 2>/dev/null; then
  echo "tmux session '$name' already exists; attach with: tmux attach -t $name"
  exit 0
fi
setsid -f tmux new-session -d -s "$name" \
  "uv run python scripts/ablate.py '$grid' $* > 'runs/$name/ablate.log' 2>&1"
sleep 2
tmux ls
echo "launched '$name'; log: runs/$name/ablate.log"
