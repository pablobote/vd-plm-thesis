# coding=utf-8
# Simplified from RevisitVD/finetune/Finetune_LLMs.py
#
# Supports three models, all fine-tunable on Colab Pro (H100/A100) via QLoRA:
#
#   --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct   ← best results, in the paper
#   --model_name_or_path codellama/CodeLlama-7b-Instruct-hf          ← also in the paper, Meta's model
#   --model_name_or_path microsoft/phi-2                             ← smaller (2.7B), faster, good for testing
#
# What this script does differently from the original:
#   - Removed DeepSpeed / ZeRO (overkill for single-GPU Colab, and requires a separate config file)
#   - Removed the --resume logic (fragile and not needed for a clean run)
#   - Removed --ft_head (head-only finetuning is dominated by LoRA in every benchmark)
#   - Removed --run_dir and TensorBoard hooks
#   - Removed dist.destroy_process_group() call (not needed without DeepSpeed)
#   - Simplified accelerator usage — still present for future multi-GPU, but transparent for single GPU
#   - Made --lora and --q always True (there is no reason to run a 7B model without them on Colab)
#   - Added --checkpoint_path for convenient test-only runs (same pattern as V_Simple_Finetune_SLMs.py)
#   - Added console logging so you see output in the notebook, not just in the log file
#
# Required installs (run once in your Colab notebook):
#   pip install transformers peft accelerate bitsandbytes scikit-learn tqdm

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import gc
import json
import random
import re
import numpy as np

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from accelerate import Accelerator
from accelerate.utils import set_seed

import warnings
warnings.filterwarnings('ignore')

# ── Logging ───────────────────────────────────────────────────────────────────
# We set up a logger that writes to both a file and the console so you can see
# progress live in your Colab notebook.
logger = logging.getLogger(__name__)


# ── Data ──────────────────────────────────────────────────────────────────────

class InputFeatures:
    """Holds the tokenized representation of one code sample."""
    def __init__(self, input_ids, attention_mask, label, index):
        self.input_ids    = input_ids       # tensor of token IDs, shape [1, max_length]
        self.attention_mask = attention_mask  # 1 for real tokens, 0 for padding
        self.label        = label           # 0 (safe) or 1 (vulnerable)
        self.index        = index           # original sample index from the dataset


def convert_to_features(js, tokenizer, args):
    """
    Tokenize one JSONL sample for an LLM backbone.

    LLMs use tokenize_plus (returns input_ids + attention_mask together) rather
    than the manual [CLS]/[SEP] construction used for BERT-style models, because
    decoder-based models don't have those special tokens. The tokenizer handles
    padding and truncation to max_length automatically.

    Whitespace is collapsed with ' '.join(split()) — same normalization as
    CodeBERT in the SLM script.
    """
    code = ' '.join(js['func'].split())

    encoded = tokenizer(
        code,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",   # returns torch tensors directly
    )
    return InputFeatures(
        encoded["input_ids"],
        encoded["attention_mask"],
        js['target'],
        js['idx'],
    )


class VulnDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        with open(file_path) as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(file_path)}"):
                js = json.loads(line.strip())
                self.examples.append(convert_to_features(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Use view(-1) then slice to guarantee exactly [max_length] shape,
        # regardless of how many extra dimensions return_tensors="pt" added.
        return (
            self.examples[i].input_ids.view(-1)[:self.examples[i].input_ids.numel()],
            self.examples[i].attention_mask.view(-1)[:self.examples[i].attention_mask.numel()],
            torch.tensor(self.examples[i].label),
        )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds):
    acc    = accuracy_score(labels, preds)
    prec   = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1     = f1_score(labels, preds, zero_division=0)
    TN, FP, FN, TP = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tnr  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr  = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    bacc = (recall + tnr) / 2
    s = lambda x: round(x, 4) * 100
    return s(acc), s(prec), s(recall), s(f1), s(tnr), s(fpr), s(fnr), s(bacc)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, accelerator, model, tokenizer, train_dataset, eval_dataset):
    """
    Fine-tune the model using QLoRA.

    Key concepts used here:

    QLoRA (Quantized Low-Rank Adaptation):
      The base model weights are frozen and loaded in 4-bit to save VRAM. A small
      set of trainable low-rank adapter matrices (LoRA) are injected into every
      linear layer. Only the adapter weights are updated during training. This lets
      you fine-tune a 7B model on a single GPU that couldn't even load the full
      model at full precision.

    Gradient accumulation:
      If your batch size is 4 and gradient_accumulation_steps is 4, the optimizer
      steps only every 4 batches, effectively simulating a batch size of 16 without
      needing 4x the VRAM.

    Accelerator:
      HuggingFace Accelerate abstracts away device placement and mixed precision.
      On a single GPU it's mostly transparent — it just ensures tensors and the
      model are on the right device.
    """
    train_loader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size, num_workers=2, pin_memory=False
    )
    eval_loader = DataLoader(
        eval_dataset, sampler=SequentialSampler(eval_dataset),
        batch_size=args.eval_batch_size, num_workers=2, pin_memory=False
    )

    total_steps  = args.epoch * len(train_loader)
    warmup_steps = int(total_steps * 0.1) if args.warmup_steps == -1 else args.warmup_steps

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0},
    ], lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Accelerator wraps model, dataloaders, optimizer, and scheduler so they
    # work correctly regardless of the underlying device/precision setup.
    model, train_loader, eval_loader, optimizer, scheduler = accelerator.prepare(
        model, train_loader, eval_loader, optimizer, scheduler
    )

    logger.info(f"Training: {len(train_dataset)} examples, {args.epoch} epochs, "
                f"batch={args.train_batch_size}, grad_accum={args.gradient_accumulation_steps}")

    best_bacc = 0.0
    model.zero_grad()

    for epoch in range(args.epoch):
        model.train()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}",
                   disable=not accelerator.is_local_main_process)

        for step, (input_ids, attention_mask, labels) in enumerate(bar):
            with accelerator.accumulate(model):
                # Forward pass — AutoModelForSequenceClassification returns
                # (loss, logits) when labels are provided.
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss   = output.loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            bar.set_postfix(loss=round(loss.item(), 4))

        # Evaluate after every epoch
        results = evaluate(args, accelerator, model, eval_dataset, eval_loader)
        for k, v in results.items():
            logger.info(f"  {k} = {round(v, 4)}")

        # Save best checkpoint
        if results['eval_bacc'] > best_bacc:
            best_bacc = results['eval_bacc']
            save_dir = os.path.join(
                args.output_dir, args.localtime, args.project, 'checkpoint-best-bacc'
            )
            os.makedirs(save_dir, exist_ok=True)

            accelerator.wait_for_everyone()
            # Save the LoRA adapter weights only (not the frozen base model —
            # that stays on HuggingFace and is loaded fresh at test time).
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

            logger.info(f"  *** New best bacc {best_bacc:.4f} — adapter saved to {save_dir}")

    accelerator.wait_for_everyone()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(args, accelerator, model, dataset, dataloader):
    """
    Run the model on the validation set and compute all metrics.

    Note on softmax: LLMs used for classification output raw logits over
    num_labels classes (here 2: safe and vulnerable). We apply softmax and
    take the probability of class 1 (vulnerable), thresholding at 0.5.
    This is different from the SLM script which uses sigmoid on a single
    logit — here we have two output neurons because AutoModelForSequenceClassification
    with num_labels=2 outputs a 2-class head.
    """
    model.eval()
    all_probs, all_labels = [], []
    total_loss, steps = 0.0, 0

    bar = tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
    for input_ids, attention_mask, labels in bar:
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss, logits = output.loss, output.logits
        logits, labels = accelerator.gather_for_metrics((logits, labels))

        prob = torch.nn.functional.softmax(logits, dim=-1)
        all_probs.append(prob.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.mean().item()
        steps += 1

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = probs[:, 1] > 0.5   # class 1 = vulnerable

    acc, prec, recall, f1, tnr, fpr, fnr, bacc = compute_metrics(labels, preds)
    return {
        'eval_loss': total_loss / steps,
        'eval_acc': acc, 'eval_prec': prec, 'eval_recall': recall,
        'eval_f1': f1,   'eval_tnr': tnr,   'eval_fpr': fpr,
        'eval_fnr': fnr, 'eval_bacc': bacc,
    }


# ── Test ──────────────────────────────────────────────────────────────────────

def test(args, accelerator, model, dataset, dataloader):
    model.eval()
    all_probs, all_labels = [], []

    bar = tqdm(dataloader, desc="Testing", disable=not accelerator.is_local_main_process)
    for input_ids, attention_mask, labels in bar:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        logits, labels = accelerator.gather_for_metrics((logits, labels))
        prob = torch.nn.functional.softmax(logits, dim=-1)
        all_probs.append(prob.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = probs[:, 1] > 0.5

    acc, prec, recall, f1, tnr, fpr, fnr, bacc = compute_metrics(labels, preds)
    result = {
        'test_acc': acc, 'test_prec': prec, 'test_recall': recall,
        'test_f1': f1,   'test_tnr': tnr,   'test_fpr': fpr,
        'test_fnr': fnr, 'test_bacc': bacc,
    }

    if accelerator.is_main_process:
        out_dir = os.path.join(args.output_dir, args.localtime, args.project, 'bacc')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'predictions.txt'), 'w') as f:
            for ex, pred in zip(dataset.examples, preds):
                f.write(f"idx: {ex.index}, pred: {int(pred)}, target: {ex.label}\n")
        np.savez(os.path.join(out_dir, 'result.npz'), test_result=result)

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_qlora_config(args):
    """
    4-bit NF4 quantization config.
    NF4 (NormalFloat4) is a data type designed for normally distributed weights —
    which transformer weights typically are. Combined with double quantization
    (quantizing the quantization constants themselves), this reduces a 7B model
    from ~28GB (fp32) to ~4GB, making it fit on a single 16GB T4 or 40GB A100.
    bfloat16 is used for the compute dtype (the actual matrix multiplications)
    because it's numerically stable and well-supported on modern GPUs/TPUs.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_lora_config(args):
    """
    LoRA adapter config.
    r (rank): controls the size of the adapter matrices. Higher = more parameters
      but also more expressiveness. 64 is a good default for VD tasks.
    lora_alpha: scaling factor. The effective learning rate of the adapter is
      lora_alpha / r. Setting alpha=16 with r=64 gives a scale of 0.25.
    target_modules='all-linear': injects adapters into every linear layer in the
      model, which gives better coverage than targeting only attention layers.
    task_type=SEQ_CLS: tells PEFT this is a sequence classification task, which
      ensures the classification head is handled correctly.
    """
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
    )


def load_base_model(args, q_config):
    """Load the base model with 4-bit quantization and set the pad token.

    For models like Phi-2, pad_token_id is not set in the config at all,
    which causes a crash during model __init__ because the architecture reads
    it before we get a chance to set it. We load the config separately first,
    patch it, then pass it into from_pretrained so the model never sees a
    missing pad_token_id.
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # Set pad_token_id before the model is built — some architectures (Phi-2)
    # read it in __init__ and crash if it's missing.
    if not getattr(config, 'pad_token_id', None):
        config.pad_token_id = config.eos_token_id

    # Set num_labels on the config rather than passing as a kwarg — newer
    # versions of transformers no longer accept num_labels in from_pretrained.
    config.num_labels = 2
    config.use_cache = False   # must be False for gradient checkpointing

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        quantization_config=q_config,
        device_map={"": 0},
    )
    return model


def fix_model_devices(model, device):
    """
    Ensure every parameter/buffer ends up on `device`. Non-persistent buffers
    (e.g. Phi3's rotary-embedding inv_freq) aren't part of the checkpoint state
    dict, so device_map/low_cpu_mem_usage loading can silently leave them on
    CPU even though every real weight landed on GPU.

    Phi3RotaryEmbedding also stashes a copy as `self.original_inv_freq` — a
    plain Python attribute, NOT a registered buffer — so neither model.to()
    nor named_buffers() ever touches it. Its "longrope" scaling resets
    `inv_freq` from this stale, never-moved copy on every forward call for
    sequences shorter than the model's original context length (true for our
    max_length), silently undoing any buffer fix. A forward pre-hook re-syncs
    both the buffer and this attribute immediately before every forward call.
    """
    model.to(device)  # recursive move; safe for bitsandbytes params too

    def _sync_module_tensors(module, _args=None):
        for _, buf in module.named_buffers(recurse=False):
            if buf.device != device:
                buf.data = buf.data.to(device)
        orig = getattr(module, "original_inv_freq", None)
        if isinstance(orig, torch.Tensor) and orig.device != device:
            module.original_inv_freq = orig.to(device)

    for module in model.modules():
        _sync_module_tensors(module)
        if any(True for _ in module.named_buffers(recurse=False)) or hasattr(module, "original_inv_freq"):
            module.register_forward_pre_hook(_sync_module_tensors)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()

    # Required
    p.add_argument('--project',            type=str, required=True,
                   help="Run name, used to organise output folders.")
    p.add_argument('--model_name_or_path', type=str, required=True,
                   help="HuggingFace model id. Recommended: deepseek-ai/deepseek-coder-6.7b-instruct "
                        "| codellama/CodeLlama-7b-Instruct-hf | microsoft/phi-2")
    p.add_argument('--output_dir',         type=str, required=True)

    # Data
    p.add_argument('--train_data_file', type=str, default=None)
    p.add_argument('--eval_data_file',  type=str, default=None)
    p.add_argument('--test_data_file',  type=str, default=None)
    p.add_argument('--test_project',    type=str, default=None,
                   help="Override project name in output paths for test-only runs.")

    # Flags
    p.add_argument('--do_train', action='store_true')
    p.add_argument('--do_eval',  action='store_true',
                   help="Unused — evaluation runs automatically after every epoch during training.")
    p.add_argument('--do_test',  action='store_true')
    p.add_argument('--evaluate_during_training', action='store_true',
                   help="Unused — kept for CLI compatibility. Eval always runs after each epoch.")

    # Hyperparameters
    p.add_argument('--max_length',                  type=int,   default=512)
    p.add_argument('--epoch',                       type=int,   default=10)
    p.add_argument('--train_batch_size',            type=int,   default=4)
    p.add_argument('--eval_batch_size',             type=int,   default=4)
    p.add_argument('--test_batch_size',             type=int,   default=4)
    p.add_argument('--gradient_accumulation_steps', type=int,   default=4,
                   help="Accumulate gradients over N batches before stepping. "
                        "Effective batch size = train_batch_size * gradient_accumulation_steps.")
    p.add_argument('--learning_rate',               type=float, default=2e-5)
    p.add_argument('--weight_decay',                type=float, default=0.0)
    p.add_argument('--adam_epsilon',                type=float, default=1e-8)
    p.add_argument('--max_grad_norm',               type=float, default=1.0)
    p.add_argument('--warmup_steps',                type=int,   default=-1,
                   help="-1 = auto (10%% of total steps)")
    p.add_argument('--seed',                        type=int,   default=42)

    # LoRA / QLoRA
    p.add_argument('--lora_rank',    type=int,   default=64)
    p.add_argument('--lora_alpha',   type=int,   default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)

    # Checkpoint
    p.add_argument('--localtime',        type=str, default='run')
    p.add_argument('--basetime',         type=str, default='run',
                   help="Timestamp of a previous training run, used to find its checkpoint for --do_test only.")
    p.add_argument('--checkpoint_path',  type=str, default=None,
                   help="Direct path to a saved LoRA adapter directory. Overrides localtime/basetime resolution.")

    args = p.parse_args()

    # Accelerator — single GPU on Colab, this is mostly transparent
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps if args.do_train else 1
    )

    # Logging to file + console
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f'logs/{args.localtime}.log', mode='w'),
            logging.StreamHandler(),
        ]
    )

    set_seed(args.seed)
    logger.info(f"Args: {args}")

    # Build configs — always use QLoRA, no option to disable
    q_config   = build_qlora_config(args)
    lora_config = build_lora_config(args)

    # Tokenizer — shared between train and test
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Train ────────────────────────────────────────────────────────────────
    if args.do_train:
        if not args.train_data_file:
            p.error("--train_data_file is required when --do_train is set.")
        if not args.eval_data_file:
            p.error("--eval_data_file is required when --do_train is set.")

        model = load_base_model(args, q_config)
        # prepare_model_for_kbit_training enables gradient checkpointing and
        # casts non-quantized layers (like layer norms) to float32 for stability.
        model = prepare_model_for_kbit_training(model)
        # Wrap the base model with LoRA adapter layers
        model = get_peft_model(model, lora_config)
        fix_model_devices(model, accelerator.device)
        model.print_trainable_parameters()  # useful sanity check — should be ~1-2% of total

        train_dataset = VulnDataset(tokenizer, args, args.train_data_file)
        eval_dataset  = VulnDataset(tokenizer, args, args.eval_data_file)
        train(args, accelerator, model, tokenizer, train_dataset, eval_dataset)

    # Free VRAM before test
    if args.do_train:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Test ─────────────────────────────────────────────────────────────────
    if args.do_test:
        if not args.test_data_file:
            p.error("--test_data_file is required when --do_test is set.")

        # Resolve checkpoint — same three-way logic as V_Simple_Finetune_SLMs.py
        if args.checkpoint_path:
            adapter_dir = args.checkpoint_path
        elif args.do_train:
            adapter_dir = os.path.join(
                args.output_dir, args.localtime, args.project, 'checkpoint-best-bacc'
            )
        else:
            adapter_dir = os.path.join(
                args.output_dir, args.basetime, args.project, 'checkpoint-best-bacc'
            )

        if not os.path.exists(adapter_dir):
            logger.error(f"Checkpoint not found: {adapter_dir}")
            logger.error("Tip: use --checkpoint_path to point directly to the adapter folder.")
            return

        logger.info(f"Loading adapter from: {adapter_dir}")

        # Load base model fresh, then inject the saved LoRA weights on top
        base_model = load_base_model(args, q_config)
        base_model = prepare_model_for_kbit_training(base_model)
        model      = PeftModel.from_pretrained(base_model, adapter_dir)
        fix_model_devices(model, accelerator.device)

        test_dataset = VulnDataset(tokenizer, args, args.test_data_file)
        test_loader  = DataLoader(
            test_dataset, sampler=SequentialSampler(test_dataset),
            batch_size=args.test_batch_size, num_workers=2, pin_memory=False
        )
        model, test_loader = accelerator.prepare(model, test_loader)

        result = test(args, accelerator, model, test_dataset, test_loader)
        logger.info("***** Test results *****")
        for k, v in sorted(result.items()):
            logger.info(f"  {k} = {round(v, 4)}")


if __name__ == "__main__":
    main()
