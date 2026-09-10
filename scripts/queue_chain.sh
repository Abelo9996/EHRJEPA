#!/usr/bin/env bash
# Queue a grid to start after ALL currently running or queued grid units finish.
#
#   scripts/queue_chain.sh <grid.yaml> <run-name>
#
# Waits until no `scripts/ablate.py` runner and no `queue_after.sh` wrapper is
# alive, then launches the grid via launch_grid.sh. Runs as a systemd unit as
# root so it survives ssh sessions.
set -euo pipefail
cd "$(dirname "$0")/.."
grid="$1"; name="$2"
mkdir -p "runs/$name"
repo="$PWD"; owner="${GRID_USER:-$(stat -c %U .)}"; log="$repo/runs/$name/queue.log"
unit="chain-$name"
if systemctl is-active --quiet "$unit"; then echo "unit $unit already active"; exit 0; fi
chown -R "$owner" "runs/$name"; touch "$log"
systemd-run --unit="$unit" --collect -p WorkingDirectory="$repo" \
  -p StandardOutput="append:$log" -p StandardError="append:$log" \
  /bin/bash -lc "while pgrep -f 'bin/python3? scripts/ablate.py' >/dev/null || pgrep -f 'queue_after.sh' >/dev/null; do sleep 120; done; scripts/launch_grid.sh '$grid' '$name'"
sleep 2; systemctl is-active "$unit" && echo "chained '$name' after all running/queued grids; log: runs/$name/queue.log"
