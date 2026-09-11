#!/usr/bin/env bash
# Relaunch the a2 pipeline on the GPU host (run as root inside WSL).
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/launch_grid.sh configs/grids/a2_physionet2019.yaml a2-physionet2019 2>&1 | tail -n 1
sleep 20
scripts/queue_grid.sh a2_physionet2019 configs/grids/a2_physionet2012.yaml a2-physionet2012 2>&1 | tail -n 1
scripts/queue_chain.sh configs/grids/a2ft_physionet2019.yaml a2ft-physionet2019 2>&1 | tail -n 1
scripts/queue_chain.sh configs/grids/a2ft_physionet2012.yaml a2ft-physionet2012 2>&1 | tail -n 1
