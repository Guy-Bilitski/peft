import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from tqdm import tqdm
import evaluate
import numpy as np
from peft import UIOrthoLoRAConfig, get_peft_model, TaskType, PeftConfig, PeftModel
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset
from numbers import Number
from pathlib import Path
import json
import numpy as np
from transformers.trainer_utils import EvalPrediction
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader
import datetime

def load_and_prepare(tokenizer, max_length=128):
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
    return ds



class CiderMetric:
    """Wraps pycocoevalcap so it looks like an `evaluate` metric."""
    def __init__(self):
        self.scorer = Cider()

    def compute(self, *, predictions, references):
        # pycocoevalcap expects dicts: {idx: ["sentence"]}
        hyps = {i: [pred] for i, pred in enumerate(predictions)}
        refs = {i: [ref]  for i, ref  in enumerate(references)}
        score, _ = self.scorer.compute_score(refs, hyps)
        return {"cider": score}

cider_metric = CiderMetric()

bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")
nist_metric = evaluate.load("nist_mt")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels


def set_contiguous(model):
    for m in model.modules():
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            base = m.parametrizations.weight[0].base
            if not base.is_contiguous():
                base.data = base.data.contiguous()

def compute_metrics(eval_pred):
    """Compute BLEU, METEOR, ROUGE-L, (optionally) NIST on E2E-NLG."""
    preds, labels = eval_pred

    # ── tensors → numpy ───────────────────────────────────────────────
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # ── logits → ids if necessary ─────────────────────────────────────
    if preds.ndim == 3:                      # (batch, seq, vocab)
        preds = preds.argmax(-1)

    # ── un-mask labels ────────────────────────────────────────────────
    labels = labels.copy()
    labels[labels == -100] = tokenizer.pad_token_id

    # ── decode ────────────────────────────────────────────────────────
    pred_strs  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    pred_strs  = [s.strip() for s in pred_strs]
    label_strs = [s.strip() for s in label_strs]

    # ── metrics ───────────────────────────────────────────────────────
    bleu   = bleu_metric.compute(
                predictions=pred_strs,
                references=[[r] for r in label_strs]
             )["score"]

    meteor = meteor_metric.compute(
                predictions=pred_strs,
                references=label_strs
             )["meteor"]

    rougeL = rouge_metric.compute(
                predictions=pred_strs,
                references=label_strs,
                use_stemmer=True
             )["rougeL"]

    cider = cider_metric.compute(
                predictions=pred_strs,
                references=label_strs   # wrapper handles dict-conversion
            )["cider"]

    # ---- NIST (may not exist on tiny samples) ------------------------
    nist_raw = nist_metric.compute(
                predictions=pred_strs,
                references=[[r] for r in label_strs]
              )
    nist_val = nist_raw.get("nist_mt")

    # helper to round only numerics
    def _r(x):
        return round(float(x), 4) if isinstance(x, (int, float)) else x

    out = {
        "bleu"  : _r(bleu),
        "meteor": _r(meteor),
        "rougeL": _r(rougeL),
        "cider"  : _r(cider),
    }
    if nist_val is not None:
        out["nist"] = _r(nist_val)

    return out


orthoLoRAConfig = UIOrthoLoRAConfig(
    target_modules=["attn.c_attn", "attn.c_proj"],
    fan_in_fan_out         = True,   # GPT-2 matrices are (out, in)
    initial_scaler         = 0.1,    # scale of the diagonal Σ at init
    initial_sigma          = 0.1,    # std-dev for the trainable Σ entries
    uiortholora_alpha      = 1,
    uiortholora_dropout    = 0,
    num_svalues_to_adapt   = 2,       # adapt the top-4 singular values
    num_svectors_to_adapt  = 2,       # adapt the corresponding vectors
    task_type              = TaskType.CAUSAL_LM
)


model_path = "gpt2-medium"
seed=42

torch.manual_seed(seed)
np.random.seed(seed)


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_and_prepare(tokenizer)

base_model = AutoModelForCausalLM.from_pretrained(model_path)
base_model.config.pad_token_id = tokenizer.pad_token_id

model = get_peft_model(base_model, orthoLoRAConfig)

set_contiguous(model)
model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="outputs/check",
    overwrite_output_dir=True,
    eval_strategy="no",
    save_strategy="no",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    eval_accumulation_steps=2,
    learning_rate=1e-3,
    lr_scheduler_type="linear",
    label_smoothing_factor=0.1,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=50,
    save_total_limit=1,
    report_to="none",
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"].select(range(100)),
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

# # trainer.train()
# from accelerate import Accelerator
# accelerator = Accelerator()
# trainer = accelerator.prepare(trainer)
# trainer.train()


# trainer.save_model("outputs/models")

# Load the trained model from saved checkpoint
# Load config to get base model info
peft_config = PeftConfig.from_pretrained("outputs/models")

# Load base model (e.g., GPT2-medium)
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)

# Load the full adapted model (base + adapter weights)
model = PeftModel.from_pretrained(base_model, "outputs/models")

trainer.model = model


# raw = trainer.predict(ds["test"])

model.eval()
model.cuda()  # move to GPU if not already
tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

gen_preds = []
true_labels = []

dataloader = DataLoader(ds["test"].select(range(100)), batch_size=32, collate_fn=data_collator)

for batch in tqdm(dataloader, desc="Generating outputs"):
    input_ids = batch["input_ids"].cuda()
    attention_mask = (input_ids != tokenizer.pad_token_id).long().cuda()
    labels = batch["labels"]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_beams=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_preds.append(outputs.cpu())
    true_labels.append(labels)

# Stack tensors
gen_preds = torch.cat(gen_preds, dim=0)
true_labels = torch.cat(true_labels, dim=0)

# Compute and save metrics
metrics = compute_metrics((gen_preds, true_labels))

# Add timestamp and training details
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
metrics.update({
    "timestamp": timestamp,
    "base_model": peft_config.base_model_name_or_path,
    "num_training_examples": len(ds["train"]),
    "num_test_examples": len(ds["test"]),
    "num_training_epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "batch_size": training_args.per_device_train_batch_size,
    "num_beams": 4,
    "max_new_tokens": 64,
    "num_svalues_to_adapt": orthoLoRAConfig.num_svalues_to_adapt,
    "num_svectors_to_adapt": orthoLoRAConfig.num_svectors_to_adapt,
    "uiortholora_alpha": orthoLoRAConfig.uiortholora_alpha,
    "uiortholora_dropout": orthoLoRAConfig.uiortholora_dropout,
    "initial_scaler": orthoLoRAConfig.initial_scaler,
    "initial_sigma": orthoLoRAConfig.initial_sigma,
    "target_modules": list(orthoLoRAConfig.target_modules) if orthoLoRAConfig.target_modules is not None else None,  # Convert to list if it's a set
})

Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
metrics_file = Path(training_args.output_dir) / "test_metrics.json"

# Load existing metrics if file exists
existing_metrics = []
if metrics_file.exists():
    with open(metrics_file) as f:
        existing_metrics = json.load(f)
        if not isinstance(existing_metrics, list):
            existing_metrics = [existing_metrics]

# Append new metrics
existing_metrics.append(metrics)

# Helper function to convert sets to lists for JSON serialization
def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    else:
        return obj

# Convert any sets to lists before JSON serialization
existing_metrics = convert_sets_to_lists(existing_metrics)

# Write updated metrics
metrics_file.write_text(json.dumps(existing_metrics, indent=2))
print(f"Test metrics saved to {training_args.output_dir} at {timestamp}")

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


