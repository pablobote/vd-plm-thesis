#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_studies_base.py"
cd "${PROJECT_ROOT}"
if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
MODELS=("mistral_it" "qwen25_it" "phi35_it" "llama31_it")
RUN_SEEDS=(42 43 44)
INPUT_DATASETS=(
  "${PROJECT_ROOT}/perturbed_dataset/rq2/rq2a/data/ood_benchmark_rq2a.jsonl"
)
BATCH_SIZE=64; MAX_ATTEMPTS=5; MAX_TOKENS=32; SAVE_EVERY=50
for seed in "${RUN_SEEDS[@]}"; do
  for model_id in "${MODELS[@]}"; do
    for input_path in "${INPUT_DATASETS[@]}"; do
      echo "Running RQ2a model=${model_id} seed=${seed} dataset=$(basename ${input_path})"
      "${PYTHON_BIN}" "${RUNNER_SCRIPT}" \
        --model_id "${model_id}" --input_path "${input_path}" \
        --batch_size "${BATCH_SIZE}" --max_attempts "${MAX_ATTEMPTS}" \
        --max_tokens "${MAX_TOKENS}" --seed "${seed}" --save_every "${SAVE_EVERY}"
    done
  done
done
