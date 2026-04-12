#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/rebuttal_runs/glue_deberta}"
RESULT_TAG="${RESULT_TAG:-deberta_rebuttal}"
SEEDS="${SEEDS:-13,21,42}"
SUITE_LEVEL="${SUITE_LEVEL:-fast}"
ADD_POSITION="${ADD_POSITION:-befdot}"

if [[ -z "${TASKS:-}" ]]; then
  if [[ "$SUITE_LEVEL" == "full" ]]; then
    TASKS="cola sst2 mrpc stsb qqp qnli rte wnli mnli"
  else
    TASKS="cola mrpc"
  fi
fi

if [[ -z "${PRESETS:-}" ]]; then
  if [[ "$SUITE_LEVEL" == "full" ]]; then
    PRESETS="baseline naive_xavier_output_gelu naive_he_output_relu xavier_no_activation zero_no_activation unit_no_activation xavier_weight_gelu unit_weight_gelu unit_weight_relu"
  else
    PRESETS="baseline naive_xavier_output_gelu naive_he_output_relu xavier_no_activation zero_no_activation unit_no_activation xavier_weight_gelu unit_weight_gelu"
  fi
fi

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=' ' read -r -a TASK_ARRAY <<< "$TASKS"
IFS=' ' read -r -a PRESET_ARRAY <<< "$PRESETS"

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local task="$1"
  local preset="$2"
  local seed="$3"
  local existing_metrics_glob="$OUTPUT_ROOT/deberta/$task/$preset/${RESULT_TAG}_deberta_${task}_*/seed_${seed}/metrics.json"

  if compgen -G "$existing_metrics_glob" > /dev/null; then
    echo "[skip] task=$task preset=$preset seed=$seed"
    return
  fi

  local cmd=(
    "$PYTHON_BIN" test_gleu.py
    --result_path "${RESULT_TAG}_deberta"
    --glue_task "$task"
    --preset "$preset"
    --gpu 0
    --seed "$seed"
    --add_position "$ADD_POSITION"
    --run_output_root "$OUTPUT_ROOT"
    --skip_checkpoint_save
  )

  echo "[run] task=$task preset=$preset seed=$seed gpu=$GPU(torch cuda:0)"
  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
}

for task in "${TASK_ARRAY[@]}"; do
  for preset in "${PRESET_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
      run_one "$task" "$preset" "$seed"
    done
  done
done

echo "[done] GLUE rebuttal suite finished"
