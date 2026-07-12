#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/geoai-lightning-talk-research}"
SEEDS="${SEEDS:-1234,2026,3407}"

cd "$PROJECT_DIR"

if [[ -n "$WAIT_PID" ]]; then
  echo "Waiting for existing experiment PID $WAIT_PID before starting remaining runs."
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do
    sleep 300
  done
fi

echo "Starting supplemental FTW 5% label-budget run."
mkdir -p artifacts/ftw_l4_5pct_v1
uv run python scripts/run_foundation_experiment.py \
  --run-label ftw_l4_5pct_v1 \
  --datasets ftw_africa \
  --models prithvi_eo_v2_300m,terramind_v1_base,terramind_v1_large \
  --adapters linear_probe,lora \
  --label-budgets 0.05 \
  --seeds "$SEEDS" \
  --batch-size 1 \
  --image-size 224 \
  --max-train-batches full \
  --max-val-batches full \
  --device cuda

echo "Starting Ghana flood stress-test run."
mkdir -p artifacts/ghana_l4_full_v1
uv run python scripts/run_foundation_experiment.py \
  --run-label ghana_l4_full_v1 \
  --datasets sen1floods11_ghana \
  --models prithvi_eo_v2_300m,terramind_v1_base,terramind_v1_large \
  --adapters linear_probe,lora \
  --label-budgets 0.05,0.10,0.25,1.00 \
  --seeds "$SEEDS" \
  --batch-size 1 \
  --image-size 224 \
  --max-train-batches full \
  --max-val-batches full \
  --device cuda

echo "Remaining Lightning experiments completed."
