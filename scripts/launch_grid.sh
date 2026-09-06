#!/usr/bin/env bash
# Launch an ablation grid detached so it survives SSH disconnects.
#
#   scripts/launch_grid.sh <grid.yaml> <run-name> [extra ablate.py args...]
#
# Logs to runs/<run-name>/ablate.log.  Preferred backend: a systemd transient
# unit (grid-<run-name>) run as the repo owner; this is the only mechanism that
# survives the end of a `wsl -e` session under WSL2.  Requires root (invoke via
# `wsl -u root`).  Falls back to a detached tmux session when systemd is absent.
# Re-running the same command after a disconnect resumes the grid (the runner
# skips cells that already have a summary row).
set -euo pipefail
cd "$(dirname "$0")/.."
grid="$1"; name="$2"; shift 2
mkdir -p "runs/$name"
repo="$PWD"
owner="${GRID_USER:-$(stat -c %U .)}"
log="$repo/runs/$name/ablate.log"
cmd="uv run python scripts/ablate.py '$grid' $*"

if [ "$(id -u)" = 0 ] && systemctl is-system-running --quiet 2>/dev/null; then
  unit="grid-$name"
  if systemctl is-active --quiet "$unit"; then
    echo "unit $unit already active"; exit 0
  fi
  chown -R "$owner" "runs/$name"
  systemd-run --unit="$unit" --collect \
    -p User="$owner" -p WorkingDirectory="$repo" \
    -p StandardOutput="append:$log" -p StandardError="append:$log" \
    /bin/bash -lc "$cmd"
  sleep 2
  systemctl is-active "$unit" && echo "launched unit $unit; log: runs/$name/ablate.log"
else
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux session '$name' already exists; attach with: tmux attach -t $name"
    exit 0
  fi
  setsid -f tmux new-session -d -s "$name" "$cmd > '$log' 2>&1"
  sleep 2
  tmux ls
  echo "launched tmux session '$name'; log: runs/$name/ablate.log"
fi
