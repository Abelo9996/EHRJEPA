#!/bin/sh
# Wait for a process to exit, then run a command.
#
#   nohup scripts/queue_after.sh <pid> <command...> > <log> 2>&1 &
#
# One 16 GB M4 has one GPU, and two ablation grids sharing it is not "twice as
# slow", it is two runs whose tok/s numbers mean nothing and whose peak memory
# is somebody else's. So a grid queued behind another one polls for the first to
# finish and then execs the second in place, which keeps the reported PID the
# one worth watching.
#
# The poll is `kill -0`, i.e. "does a process with this PID exist and can I
# signal it" -- it sends nothing. A PID can in principle be recycled inside the
# 60-second window and this would then wait on a stranger; on a laptop running
# one long job that is a risk worth taking over a lock file nobody cleans up.
set -eu

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <pid> <command...>" >&2
    exit 2
fi

pid="$1"
shift

echo "[queue_after] waiting for pid $pid before: $*"
while kill -0 "$pid" 2>/dev/null; do
    sleep 60
done
echo "[queue_after] pid $pid is gone at $(date -u '+%Y-%m-%dT%H:%M:%SZ'); starting"
exec "$@"
