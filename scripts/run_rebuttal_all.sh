#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
SUITE_LEVEL="${SUITE_LEVEL:-fast}"
GLUE_OUTPUT_ROOT="${GLUE_OUTPUT_ROOT:-$ROOT_DIR/rebuttal_runs/glue_deberta}"
GEN_OUTPUT_ROOT="${GEN_OUTPUT_ROOT:-$ROOT_DIR/rebuttal_runs/generation_t5}"
VIT_OUTPUT_ROOT="${VIT_OUTPUT_ROOT:-$ROOT_DIR/rebuttal_runs/vit}"
AGGREGATE_INPUT_ROOT="${AGGREGATE_INPUT_ROOT:-$ROOT_DIR/rebuttal_runs}"
AGGREGATE_OUTPUT_PREFIX="${AGGREGATE_OUTPUT_PREFIX:-$ROOT_DIR/rebuttal_runs/aggregated_rebuttal_results}"
RUN_GLUE="${RUN_GLUE:-1}"
RUN_GEN="${RUN_GEN:-1}"
RUN_VIT="${RUN_VIT:-1}"

if [[ "$RUN_GLUE" == "1" ]]; then
  echo "[start] running GLUE suite (SUITE_LEVEL=$SUITE_LEVEL)"
  OUTPUT_ROOT="$GLUE_OUTPUT_ROOT" GPU="$GPU" SUITE_LEVEL="$SUITE_LEVEL" bash scripts/run_rebuttal_glue.sh
fi

if [[ "$RUN_GEN" == "1" ]]; then
  echo "[start] running generation suite (SUITE_LEVEL=$SUITE_LEVEL)"
  OUTPUT_ROOT="$GEN_OUTPUT_ROOT" GPU="$GPU" SUITE_LEVEL="$SUITE_LEVEL" bash scripts/run_rebuttal_generation.sh
fi

if [[ "$RUN_VIT" == "1" ]]; then
  echo "[start] running ViT suite (SUITE_LEVEL=$SUITE_LEVEL)"
  OUTPUT_ROOT="$VIT_OUTPUT_ROOT" GPU="$GPU" SUITE_LEVEL="$SUITE_LEVEL" bash scripts/run_rebuttal_vit.sh
fi

echo "[start] aggregating rebuttal metrics"
"$PYTHON_BIN" aggregate_rebuttal_results.py \
  --input_root "$AGGREGATE_INPUT_ROOT" \
  --output_prefix "$AGGREGATE_OUTPUT_PREFIX"

echo "[done] all rebuttal suites finished"
