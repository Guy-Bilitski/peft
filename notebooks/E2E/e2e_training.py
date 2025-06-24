import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
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
                references=label_strs
              )
    nist_val = nist_raw.get("nist", nist_raw.get("score"))  # could be None

    # helper to round only numerics
    def _r(x):
        return round(float(x), 4) if isinstance(x, Number) else x

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


raw = trainer.predict(ds["test"],
                      max_length=64,
                      num_beams=1,
                      per_device_eval_batch_size=8)  # reduce if still freezes

metrics = compute_metrics((raw.predictions, raw.label_ids))
Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
(Path(training_args.output_dir) / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
print("Test metrics saved to", training_args.output_dir)


for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


