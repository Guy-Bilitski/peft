#!/usr/bin/env python
"""
Fine–tune GPT‑2 on the E2E NLG benchmark with PEFT (LoRA / VeRA) and evaluate with
BLEU, NIST, METEOR, ROUGE‑L and CIDEr.  
Usage (LoRA example):
    python train_e2e_peft.py \
        --model_name_or_path gpt2-medium \
        --output_dir runs/gpt2-medium-lora \
        --peft lora \
        --num_train_epochs 3

For VeRA (requires PEFT>=0.11.0 or your custom tuner)::
    python train_e2e_peft.py --peft vera [...]

Notes
-----
* The script treats the task as *conditional generation*: the Meaning‑Representation
  (MR) string is fed as prompt, the model learns to generate the natural‑language
  utterance.
* Only the adapter parameters are updated; the base model is frozen.
* Metrics are computed on the dev set after every epoch and once on the test set
  at the end.
"""
import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
# ------ (optional) import your VeRA class ------
try:
    from peft import VeraConfig  # hypothetical name, adjust if different
except ImportError:
    VeraConfig = None

# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine‑tune GPT‑2 on E2E with PEFT")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2-medium")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--peft", type=str, choices=["lora", "vera"], default="lora")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank (ignored for VeRA)")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Dataset & preprocessing
# ---------------------------------------------------------------------------

def load_and_prepare(tokenizer, max_length):
    """Load E2E dataset and prepare tokenised fields."""
    ds = load_dataset("tuetschek/e2e_nlg")

    def linearise(record):
        mr = record["meaning_representation"]  # e.g. "name[Bibimbap House], food[Indian]"
        ref = record["human_reference"] if "human_reference" in record else record["reference"]
        prompt = f"{mr} => "  # simple prompt pattern
        example = prompt + ref
        tokenised = tokenizer(
            example,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        labels = tokenised["input_ids"].copy()
        # Mask prompt tokens so they are ignored in the loss (label = -100)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = [-100] * prompt_len
        tokenised["labels"] = labels
        return tokenised

    ds = ds.map(linearise, remove_columns=ds["train"].column_names)
    ds = ds.rename_column("train", "train") if "train" in ds else ds  # HF quirk safety
    return ds


# ---------------------------------------------------------------------------
# 3. Metric computation
# ---------------------------------------------------------------------------
bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")

try:
    nist_metric = evaluate.load("nist_mt")
except Exception:
    nist_metric = None

try:
    cider_metric = evaluate.load("cider")
except Exception:
    cider_metric = None


def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels[labels == -100] = tokenizer.pad_token_id
    preds_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    preds, refs = postprocess_text(preds_text, labels_text)

    # BLEU (SacreBLEU) expects list[str] and list[list[str]]
    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])["score"]

    meteor = meteor_metric.compute(predictions=preds, references=refs)["meteor"]
    rougeL = rouge_metric.compute(predictions=preds, references=refs, use_stemmer=True)[
        "rougeL"
    ]

    output = {"bleu": bleu, "meteor": meteor, "rougeL": rougeL}

    if nist_metric is not None:
        try:
            nist = nist_metric.compute(predictions=preds, references=refs)["score"]
            output["nist"] = nist
        except Exception:
            pass
    if cider_metric is not None:
        try:
            cider = cider_metric.compute(predictions=preds, references=refs)["cider"]
            output["cider"] = cider
        except Exception:
            pass

    return {k: round(v, 4) for k, v in output.items()}


# ---------------------------------------------------------------------------
# 4. Main training routine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_and_prepare(tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.peft == "lora":
        peft_cfg = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=[
                "c_attn",  # GPT‑2 attention projection
                "c_fc",    # MLP inner proj
            ],
            bias="none",
        )
    else:  # VeRA
        if VeraConfig is None:
            raise ImportError("VeRAConfig not found. Please install your custom PEFT fork.")
        peft_cfg = VeraConfig(
            rank=1,
            dropout=args.dropout,
            target_modules=["c_attn", "c_fc"],
        )

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="outputs",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        learning_rate=1e-3,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Final evaluation on the official test split (if provided)
    if "test" in ds:
        metrics = trainer.evaluate(ds["test"])
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("Test metrics saved to", args.output_dir)
