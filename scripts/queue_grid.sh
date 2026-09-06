#!/usr/bin/env bash
# Queue an ablation grid to start after another running grid finishes.
#
#   scripts/queue_grid.sh <wait-for-grid-yaml-substring> <grid.yaml> <run-name>
#
# Finds the PID of the running Python `scripts/ablate.py` whose command line
# contains the first argument and launches <grid.yaml> behind it via
# scripts/queue_after.sh, using the same backend as scripts/launch_grid.sh
# (systemd transient unit as root, else tmux).  Refuses to launch if nothing
# matching is running.
set -euo pipefail
cd "$(dirname "$0")/.."
wait_for="$1"; grid="$2"; name="$3"
pid="$(pgrep -f "bin/python3? scripts/ablate.py.*${wait_for}" | head -n 1 || true)"
if [ -z "$pid" ]; then
  echo "no running ablate.py matching '${wait_for}'; refusing to launch (use scripts/launch_grid.sh)" >&2
  exit 1
fi
mkdir -p "runs/$name"
repo="$PWD"
owner="${GRID_USER:-$(stat -c %U .)}"
log="$repo/runs/$name/ablate.log"
cmd="scripts/queue_after.sh $pid uv run python scripts/ablate.py '$grid'"

if [ "$(id -u)" = 0 ] && systemctl is-system-running --quiet 2>/dev/null; then
  unit="grid-$name"
  if systemctl is-active --quiet "$unit"; then
    echo "unit $unit already active"; exit 0
  fi
  chown -R "$owner" "runs/$name"; touch "$log"; chown "$owner" "$log"
  systemd-run --unit="$unit" --collect \
    -p User="$owner" -p WorkingDirectory="$repo" \
    -p StandardOutput="append:$log" -p StandardError="append:$log" \
    /bin/bash -lc "$cmd"
  sleep 2
  systemctl is-active "$unit" && echo "queued unit $unit behind PID $pid; log: runs/$name/ablate.log"
else
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux session '$name' already exists"; exit 0
  fi
  setsid -f tmux new-session -d -s "$name" "$cmd > '$log' 2>&1"
  sleep 2
  tmux ls
  echo "queued tmux session '$name' behind PID $pid; log: runs/$name/ablate.log"
fi
