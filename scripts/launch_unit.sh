#!/usr/bin/env bash
# Run an arbitrary repo command detached as a systemd transient unit (WSL2-safe).
#
#   scripts/launch_unit.sh <name> <command...>
#
# Logs to runs/<name>/unit.log; unit is named job-<name>; run as root
# (`wsl -u root`) so the unit can be created; the command runs as the repo owner.
set -euo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift
mkdir -p "runs/$name"
repo="$PWD"; owner="${GRID_USER:-$(stat -c %U .)}"; log="$repo/runs/$name/unit.log"
unit="job-$name"
if systemctl is-active --quiet "$unit"; then echo "unit $unit already active"; exit 0; fi
chown -R "$owner" "runs/$name"; touch "$log"; chown "$owner" "$log"
systemd-run --unit="$unit" --collect -p User="$owner" -p WorkingDirectory="$repo" \
  -p StandardOutput="append:$log" -p StandardError="append:$log" /bin/bash -lc "$*"
sleep 2; systemctl is-active "$unit" && echo "launched unit $unit; log: runs/$name/unit.log"
