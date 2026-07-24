#!/usr/bin/env bash
# run_all_ood.sh
# Runs all RQ experiments on the out-of-distribution (OOD) benchmark dataset.
#
# BEFORE RUNNING: generate the perturbation files with:
#   python generate_ood_perturbations.py \
#     --input ../dataset/ood_benchmark.jsonl \
#     --project_root . \
#     --n_instances 500 \
#     --max_func_chars 6000
#
# This already writes correctly-named output files (no renaming needed):
#   perturbed_dataset/rq1_rq3/data/ood_benchmark_rq1_rq3.jsonl
#   perturbed_dataset/rq2/rq2a/data/ood_benchmark_rq2a.jsonl
#   perturbed_dataset/rq2/rq2b/data/ood_benchmark_rq2b.jsonl

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

bash "${SCRIPT_DIR}/run_rq1_ood.sh"
bash "${SCRIPT_DIR}/run_rq2a_ood.sh"
bash "${SCRIPT_DIR}/run_rq2b_ood.sh"
bash "${SCRIPT_DIR}/run_rq3a_cot_ood.sh"
bash "${SCRIPT_DIR}/run_rq3b_selfreflect_ood.sh"
