# Revisiting Pre-trained Language Models for Vulnerability Detection: Fine-Tuning, Robustness, and a Conformity Extension

Artifact repository for the Master's thesis submitted to the Illinois Institute
of Technology, M.A.S. in Information Technology and Management.

## Overview

This repository has two components.

**1. Fine-tuning evaluation pipeline (`finetune/`, `dataset/`)** -- the main
contribution. Fine-tunes and evaluates 6 pre-trained language models for
source-code vulnerability detection on the PrimeVul dataset:

- CodeBERT, UniXCoder (full fine-tuning)
- DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B, Phi-3.5-mini, CodeLlama-7B (QLoRA)

Evaluation covers the in-distribution test set, robustness to code
normalization / identifier abstraction (UniXCoder), and out-of-distribution
generalizability on an OOD benchmark (post-training-cutoff NVD CVEs, not
used in fine-tuning) released with the base "Revisiting Pre-trained
Language Models for Vulnerability Detection" artifact.

**2. Conformity extension (`conformity_extension/`)** -- an exploratory
extension adapting the peer-pressure "conformity" framework from *Easier to
Mislead Than to Correct* (Qu et al., 2026) to zero-shot vulnerability
detection. Tests whether 4 instruction-tuned LLMs (Llama-3.1-8B, Mistral-7B,
Phi-3.5-mini, Qwen2.5-7B) can be swayed off a correct vulnerability judgment
by simulated peer/authority pressure, across 5 experimental conditions
(RQ1 baseline consensus x authority, RQ2a peer-count sweep, RQ2b
authority-weight sweep, RQ3a chain-of-thought, RQ3b self-reflection) on both
PrimeVul and the OOD benchmark.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   ├── reconstructed_{train,valid,test}.jsonl
│   ├── reconstructed_test_{norm,no_norm,abstract}.jsonl   robustness variants
│   └── ood_benchmark.jsonl                                OOD test set (from RevisitVD)
├── finetune/
│   ├── V_Simple_Finetune_SLMs.py      CodeBERT / UniXCoder / PDBERT training
│   ├── V_Simple_Finetune_LLMs.py      QLoRA training for the 4 LLMs
│   ├── model.py                       SLM model wrapper (used by the SLM script)
│   ├── dataset_perturbation.py        generates the norm/abstract test variants
│   ├── run_all_llms.sh                sequential QLoRA run for all 4 LLMs
│   ├── run_llm_robustness_ood.sh      OOD evaluation for the 4 LLMs
│   ├── Finetune_SLMs.yml, Finetune_LLMs.yml, deepspeed.yml   conda env specs
│   ├── zero_to_fp32.py                optional DeepSpeed ZeRO checkpoint utility
│   ├── visualize_results.ipynb        all thesis figures/tables for this component
│   └── output/                        result metrics (predictions + BACC/F1/etc.)
│                                       per model and condition -- checkpoints excluded,
│                                       see "Checkpoints" below
├── conformity_extension/
│   ├── generate_vd_perturbations.py   builds RQ1/RQ2a/RQ2b prompts from a
│   │                                   BigVul/DiverseVul-style JSONL (PrimeVul)
│   ├── generate_ood_perturbations.py  same, for the OOD benchmark
│   ├── scripts/                       experiment runners (run_rq*.sh) and the
│   │                                   Python study drivers / metrics scripts they call
│   ├── results_todos/                 aggregated per-seed metrics (rq1-rq3b);
│   │                                   raw per-instance generations excluded, see below
│   └── notebooks/                     rq1-rq3b visualization notebooks
└── figures/                           all figures referenced by the thesis text
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Fine-tuning was run on 4x A100 80G GPUs; the conformity extension's vLLM
inference was run on a single GPU. See the inline comments in
`finetune/V_Simple_Finetune_LLMs.py` and `finetune/run_llm_robustness_ood.sh`
for the exact per-script hardware/environment assumptions (e.g. `HF_HOME`
cache location) -- adjust these to your own machine.

### Fine-tuning pipeline

```bash
cd finetune
python V_Simple_Finetune_SLMs.py --project CodeBERT --model_type codebert ...
bash run_all_llms.sh                  # QLoRA fine-tune + test all 4 LLMs
bash run_llm_robustness_ood.sh        # OOD evaluation for the 4 LLMs
jupyter notebook visualize_results.ipynb
```

### Conformity extension

```bash
cd conformity_extension
python generate_vd_perturbations.py --input path/to/primevul.jsonl --studies rq1 rq2a rq2b
python generate_ood_perturbations.py --input ../dataset/ood_benchmark.jsonl --studies rq1 rq2a rq2b
bash scripts/run_rq1_ood.sh
python scripts/compute_all_metrics.py --results_dir results_todos
jupyter notebook notebooks/rq1_visualization.ipynb
```

## Checkpoints and large result files

Model checkpoints (`checkpoint-best-bacc/`, `*.safetensors`) and the raw
per-instance model generations under `conformity_extension/results_todos/`
(prediction jsonl files, ~4GB) are excluded via `.gitignore` -- they are
either too large for GitHub or not needed to reproduce the reported figures
(only the aggregated metrics CSVs are). Both are available on request:
**[add download link here]**.

## Known gaps

- The DeepSeek-Coder-6.7B out-of-distribution (`unseen_test`) run was not
  completed; `finetune/visualize_results.ipynb` skips it automatically in
  the OOD comparison chart, consistent with the other run scripts.
- `finetune/output/` merges results from two separate training runs: the
  final CodeBERT/UniXCoder numbers come from a later rerun than the
  original DeepSeek/Qwen/Phi/CodeLlama run; this is the exact set of
  results `visualize_results.ipynb` loads and reports.

## Citation

If you build on the base evaluation pipeline or dataset, please also cite
the original artifact this repository extends, "Revisiting Pre-trained
Language Models for Vulnerability Detection" -- this includes
`dataset/ood_benchmark.jsonl`, which is released with that artifact, not
collected as part of this thesis -- and, for the conformity extension,
Qu et al., "Easier to Mislead Than to Correct: Harmful and Beneficial
Revision in LLM Conformity" (arXiv:2606.01637).
