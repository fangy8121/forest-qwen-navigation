#!/usr/bin/env bash
# Fly the baseline coverage task on each map in turn, never in parallel.
# Usage: scripts/run_maps.sh TAG MAP [MAP ...]   e.g. scripts/run_maps.sh test train/020 train/021
# Results: $OUT/runs, Hydra snapshots: $OUT/hydra, logs: $OUT/logs (OUT defaults to outputs/baseline).
cd "$(dirname "$0")/.."
OUT=${OUT:-outputs/baseline}
PYTHON=${PYTHON:-.venvs/forest_qwen35/bin/python}
TAG=$1; shift
mkdir -p "$OUT/logs"
for MAP in "$@"; do
  NAME=${TAG}_${MAP/\//_}
  echo "=== $MAP $(date +%T)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON" scripts/fly_forest_coverage.py \
    task.scene.map_id=$MAP max_seconds=3000. \
    "output='$OUT/runs'" "hydra.run.dir='$OUT/hydra/\${now:%Y-%m-%d_%H-%M-%S}_$NAME'" \
    > "$OUT/logs/$NAME.log" 2>&1
  grep -a "COVERAGE_RESULT" "$OUT/logs/$NAME.log" | tail -1
done
echo "=== done $(date +%T)"
