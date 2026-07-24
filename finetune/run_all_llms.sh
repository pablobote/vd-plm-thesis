#!/usr/bin/env bash
set -euo pipefail

# Sequentially QLoRA fine-tune + test each LLM using V_Simple_Finetune_LLMs.py.
# No DeepSpeed/ZeRO here — plain single-GPU QLoRA, run directly with python.
cd "$(dirname "$0")"

train_data_file="../dataset/reconstructed_train.jsonl"
valid_data_file="../dataset/reconstructed_valid.jsonl"
test_data_file="../dataset/reconstructed_test.jsonl"

# project_name:model_name_or_path
# CodeLlama is last since it's a gated HF repo (needs access approval + huggingface-cli login).
models=(
  "Qwen2.5-Coder-7B:Qwen/Qwen2.5-Coder-7B-Instruct"
  "Phi-3.5-mini:microsoft/Phi-3.5-mini-instruct"
  "CodeLlama-7B:meta-llama/CodeLlama-7b-Instruct-hf"
)

for entry in "${models[@]}"; do
  project="${entry%%:*}"
  model_name_or_path="${entry#*:}"
  localtime=$(TZ="America/Chicago" date "+%Y-%m-%d-%H:%M:%S")

  echo "===== Fine-tuning $project ($model_name_or_path) — run $localtime ====="

  python V_Simple_Finetune_LLMs.py \
    --project "$project" \
    --model_name_or_path "$model_name_or_path" \
    --do_train --do_test \
    --train_data_file "$train_data_file" \
    --eval_data_file "$valid_data_file" \
    --test_data_file "$test_data_file" \
    --epoch 10 --max_length 512 \
    --train_batch_size 4 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --output_dir ./output \
    --localtime "$localtime"

  echo "===== Finished $project — see output/$localtime/$project/ ====="
done
