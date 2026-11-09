#!/usr/bin/env bash
# Regenerate the 40-seed multirow map set used by the baseline, with the same path, seed and
# obstacle profile for every map. Only maps that pass the generator's checks are published.
# Usage: scripts/generate_baseline_maps.sh [OUTPUT]   (default: the map_dir in cfg/forest_coverage.yaml)
# Add --dry-run through EXTRA to print the assignments only: EXTRA=--dry-run scripts/generate_baseline_maps.sh
set -e
cd "$(dirname "$0")/.."
OUTPUT=${1:-outputs/01_森林巡检仿真跑通/maps}
PYTHON=${PYTHON:-python}
run() { "$PYTHON" -m forest_generator generate-multirow --output "$OUTPUT" --max-attempts 6 $EXTRA "$@"; }
run --split train --seeds $(seq 20263000 20263009) $(seq 20263015 20263034)
run --split val --seeds $(seq 20263010 20263014)
run --split test --seeds $(seq 20263035 20263039)
