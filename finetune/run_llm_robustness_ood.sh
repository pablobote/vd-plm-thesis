#!/usr/bin/env bash
# ==============================================================================
# OOD (out-of-distribution) evaluation for the fine-tuned large LMs
# (DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B, Phi-3.5-mini, CodeLlama-7B)
#
# Scope: OOD generalizability only. Robustness perturbation testing
# (norm/no_norm/abstract) remains scoped to UniXCoder, consistent with
# Objective 3 in the thesis Introduction ("the best-performing encoder
# model"). This cuts the batch from 16 runs to 4 -- one per model.
#
# Mirrors exactly what was already done for CodeBERT/UniXCoder in cells 10-13
# of your finetuning notebook, just pointed at the LLM checkpoints and using
# V_Simple_Finetune_LLMs.py instead of V_Simple_Finetune_SLMs.py.
#
# Run this from the same `finetune/` directory you ran the SLM commands from.
# Intended to be left running unattended (e.g. inside tmux) -- a failure on
# one model/condition is logged and skipped rather than stopping the batch.
# ==============================================================================

set -uo pipefail

# ---- Send HuggingFace downloads to the large data volume, not the small
#      root disk that ~/.cache defaults to. This is almost certainly why
#      you ran out of space downloading base model shards.
export HF_HOME="/media/volume/vol-pablo-400g/hf_cache"
mkdir -p "$HF_HOME"

# ---- Activate the environment that has numpy/torch/peft/bitsandbytes etc. ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Finetune_LLMs

# Fail fast and loud if the environment isn't actually right, instead of
# burning a full 16-run batch on the same mistake again.
python -c "import numpy, torch, peft, transformers" || {
  echo "ERROR: Finetune_LLMs env is missing a required package. Fix this before continuing."
  exit 1
}
echo "Environment check passed: numpy/torch/peft/transformers all importable."

LOG_DIR="./output/robustness_ood_logs"
mkdir -p "$LOG_DIR"

# Perturbed / OOD data files -- these already exist from the SLM robustness run
# (Part 2 in your notebook), no need to regenerate them.
# Only running the OOD (unseen_test) condition for the large LMs.
# Robustness perturbation testing (norm/no_norm/abstract) remains scoped to
# UniXCoder only, consistent with Objective 3 in the Introduction, which
# already specified "the best-performing encoder model" rather than all
# models. See the thesis Development chapter for the full justification.
DATA_OOD="../dataset/ood_benchmark.jsonl"

# ---- Model name -> checkpoint directory (adapter dir, not a .bin file) ----
declare -A MODEL_NAME=(
  [DeepSeek-6.7B]="deepseek-ai/deepseek-coder-6.7b-instruct"
  [Qwen2.5-Coder-7B]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [Phi-3.5-mini]="microsoft/Phi-3.5-mini-instruct"
  [CodeLlama-7B]="codellama/CodeLlama-7b-Instruct-hf"
)

declare -A CHECKPOINT=(
  [DeepSeek-6.7B]="./output/DeepSeek-6.7B/checkpoint-best-bacc"
  [Qwen2.5-Coder-7B]="./output/Qwen2.5-Coder-7B/checkpoint-best-bacc"
  [Phi-3.5-mini]="./output/Phi-3.5-mini/checkpoint-best-bacc"
  [CodeLlama-7B]="./output/CodeLlama-7B/checkpoint-best-bacc"
)

# Smaller eval batch size for the 7B models than Phi-3.5-mini (3.8B), since
# this is pure inference (no optimizer states, no gradient accumulation) so
# you can generally go a bit higher than the train_batch_size you used.
# Lower these if you hit OOM.
declare -A EVAL_BATCH=(
  [DeepSeek-6.7B]=4
  [Qwen2.5-Coder-7B]=4
  [Phi-3.5-mini]=8
  [CodeLlama-7B]=4
)

run_test () {
  local model=$1
  local condition=$2       # norm | no_norm | abstract | unseen_test
  local data_file=$3

  local project="${model}_${condition}"
  local logfile="${LOG_DIR}/${project}.log"

  echo "=============================================="
  echo "Running: ${project}"
  echo "  data:       ${data_file}"
  echo "  checkpoint: ${CHECKPOINT[$model]}"
  echo "=============================================="

  python V_Simple_Finetune_LLMs.py \
    --project "${project}" \
    --model_name_or_path "${MODEL_NAME[$model]}" \
    --do_test \
    --test_data_file "${data_file}" \
    --checkpoint_path "${CHECKPOINT[$model]}" \
    --output_dir ./output \
    --max_length 512 \
    --eval_batch_size "${EVAL_BATCH[$model]}" \
    --localtime "" \
    > "${logfile}" 2>&1

  if [ $? -eq 0 ]; then
    echo "  -> OK, log: ${logfile}"
  else
    echo "  -> FAILED, check log: ${logfile}"
  fi
}

for model in "${!MODEL_NAME[@]}"; do
  echo ""
  echo "################################################"
  echo "# ${model}"
  echo "################################################"

  run_test "${model}" "unseen_test" "${DATA_OOD}"
done

echo ""
echo "All done. Check ${LOG_DIR}/ for per-run logs, and"
echo "./output/<model>_<condition>/bacc/result.npz for each result."
