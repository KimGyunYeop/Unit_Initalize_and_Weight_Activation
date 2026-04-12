#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/rebuttal_runs/vit}"
RESULT_TAG="${RESULT_TAG:-vit_rebuttal}"
SEEDS="${SEEDS:-13,21,42}"
SUITE_LEVEL="${SUITE_LEVEL:-fast}"
ADD_POSITION="${ADD_POSITION:-befdot}"

if [[ -z "${TASKS:-}" ]]; then
  if [[ "$SUITE_LEVEL" == "full" ]]; then
    TASKS="cifar100 imagenet-1k"
  else
    TASKS="cifar100"
  fi
fi

if [[ -z "${PRESETS:-}" ]]; then
  PRESETS="baseline naive_xavier_output_gelu unit_weight_gelu"
fi

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=' ' read -r -a TASK_ARRAY <<< "$TASKS"
IFS=' ' read -r -a PRESET_ARRAY <<< "$PRESETS"

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local task="$1"
  local preset="$2"
  local seed="$3"
  local existing_metrics_glob="$OUTPUT_ROOT/vit/$task/$preset/${RESULT_TAG}_vit_${task}_*/seed_${seed}/metrics.json"

  if compgen -G "$existing_metrics_glob" > /dev/null; then
    echo "[skip] dataset=$task preset=$preset seed=$seed"
    return
  fi

  local cmd=(
    "$PYTHON_BIN" test_image_classification.py
    --result_path "${RESULT_TAG}_vit"
    --image_classification_dataset "$task"
    --preset "$preset"
    --gpu 0
    --seed "$seed"
    --add_position "$ADD_POSITION"
    --run_output_root "$OUTPUT_ROOT"
    --skip_checkpoint_save
  )

  echo "[run] dataset=$task preset=$preset seed=$seed gpu=$GPU(torch cuda:0)"
  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
}

for task in "${TASK_ARRAY[@]}"; do
  for preset in "${PRESET_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
      run_one "$task" "$preset" "$seed"
    done
  done
done

echo "[done] ViT rebuttal suite finished"
