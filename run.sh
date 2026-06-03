#!/bin/bash
# Run the VIO benchmark container.
# Results are written to ./results/ on the host.
# config.yaml is mounted from the host so you can edit it without rebuilding.
#
# Usage:
#   ./run.sh                          # run with default config
#   ./run.sh python3 main.py my.yaml  # pass custom config path (must be in /vio_benchmark/)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SCRIPT_DIR/results"

docker run --rm -it \
    --network host \
    -v "$SCRIPT_DIR:/vio_benchmark" \
    vio_benchmark "$@"
