#!/usr/bin/env bash
# RQ3a — Chain-of-Thought intervention on the RQ1 dataset

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_studies_cot.py"
cd "${PROJECT_ROOT}"

if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

MODELS=("mistral_it" "qwen25_it" "phi35_it" "llama31_it")

RUN_SEEDS=(42 43 44)

INPUT_DATASETS=(
  "${PROJECT_ROOT}/perturbed_dataset/rq1_rq3/data/"*.jsonl
)

BATCH_SIZE=64
MAX_ATTEMPTS=5
MAX_TOKENS=512   # CoT needs room for step-by-step reasoning
SAVE_EVERY=50

if [ "${#INPUT_DATASETS[@]}" -eq 0 ]; then
  echo "No RQ1/RQ3 input datasets found under ${PROJECT_ROOT}/perturbed_dataset/rq1_rq3" >&2
  exit 1
fi

for seed in "${RUN_SEEDS[@]}"; do
  for model_id in "${MODELS[@]}"; do
    for input_path in "${INPUT_DATASETS[@]}"; do
      task_name="$(basename "${input_path}" .jsonl)"
      echo "Running RQ3a (CoT) model=${model_id} seed=${seed} dataset=${task_name}"
      "${PYTHON_BIN}" "${RUNNER_SCRIPT}" \
        --model_id "${model_id}" \
        --input_path "${input_path}" \
        --batch_size "${BATCH_SIZE}" \
        --max_attempts "${MAX_ATTEMPTS}" \
        --max_tokens "${MAX_TOKENS}" \
        --seed "${seed}" \
        --save_every "${SAVE_EVERY}"
    done
  done
done
