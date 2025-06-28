import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from tqdm import tqdm
import evaluate
import numpy as np
from peft import UIOrthoLoRAConfig, get_peft_model, TaskType, PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset
from pathlib import Path
import json
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader
import datetime

def load_and_prepare(tokenizer, max_length=128):
    """Load E2E dataset and prepare tokenised fields."""
    ds = load_dataset("tuetschek/e2e_nlg", trust_remote_code=True)

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

def compute_metrics(eval_pred, tokenizer):
    """Compute BLEU, METEOR, ROUGE-L, (optionally) NIST on E2E-NLG."""
    preds, labels = eval_pred

    cider_metric = CiderMetric()
    bleu_metric = evaluate.load("sacrebleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")
    nist_metric = evaluate.load("nist_mt")

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
    out = {}
    
    if bleu_metric is not None:
        bleu = bleu_metric.compute(
                    predictions=pred_strs,
                    references=[[r] for r in label_strs]
                 )["score"]
        out["bleu"] = round(float(bleu), 4)

    if meteor_metric is not None:
        meteor = meteor_metric.compute(
                    predictions=pred_strs,
                    references=label_strs
                 )["meteor"]
        out["meteor"] = round(float(meteor), 4)

    if rouge_metric is not None:
        rougeL = rouge_metric.compute(
                    predictions=pred_strs,
                    references=label_strs,
                    use_stemmer=True
                 )["rougeL"]
        out["rougeL"] = round(float(rougeL), 4)

    cider = cider_metric.compute(
                predictions=pred_strs,
                references=label_strs   # wrapper handles dict-conversion
            )["cider"]
    out["cider"] = round(float(cider), 4)

    # ---- NIST (may not exist on tiny samples) ------------------------
    if nist_metric is not None:
        nist_raw = nist_metric.compute(
                    predictions=pred_strs,
                    references=[[r] for r in label_strs]
                  )
        nist_val = nist_raw.get("nist_mt")
        if nist_val is not None:
            out["nist"] = round(float(nist_val), 4)

    return out


def get_tokenizer_and_model(model_path: str, device):
    """
    Load a base model and inject a saved PEFT adapter from `model_path`.
    """
    # 1) Load the adapter config to get the original base model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    # 2) Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model = base_model.to(device)

    # 3) Load the adapter into the base model
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)

    # 4) Ensure contiguous weights (optional)
    set_contiguous(model)

    return tokenizer, model, peft_config



def get_tokenizer(model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def finetune_model(tokenizer,training_args, orthoLoRA_args, ds, device, model_path="outputs/models", model_type="gpt2-medium"):
    print("finetuning model \n", flush=True)
    orthoLoRAConfig = UIOrthoLoRAConfig(
    target_modules=orthoLoRA_args.target_modules,
    fan_in_fan_out         = True,   # GPT-2 matrices are (out, in)
    initial_scaler         = orthoLoRA_args.initial_scaler,    # scale of the diagonal Σ at init
    initial_sigma          = orthoLoRA_args.initial_sigma,    # std-dev for the trainable Σ entries
    uiortholora_alpha      = orthoLoRA_args.uiortholora_alpha,
    uiortholora_dropout    = orthoLoRA_args.uiortholora_dropout,
    num_svalues_to_adapt   = orthoLoRA_args.num_svalues_to_adapt,       
    num_svectors_to_adapt  = orthoLoRA_args.num_svectors_to_adapt,       
    task_type              = TaskType.CAUSAL_LM)

    base_model = AutoModelForCausalLM.from_pretrained(model_type)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model = base_model.to(device)
    print("base model loaded \n", flush=True)

    orthoLora_model = get_peft_model(base_model, orthoLoRAConfig)
    set_contiguous(orthoLora_model)
    orthoLora_model = orthoLora_model.to(device)
    orthoLora_model.print_trainable_parameters()
    print("orthoLora model loaded \n", flush=True)
    
    trainer = Trainer(
        model=orthoLora_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"])

    trainer.train()
    trainer.save_model(model_path)
    print("model saved to ", model_path, flush=True)

    return trainer.model


def evaluate_model(model, tokenizer, ds, data_collator, peft_config, training_args, inference_args):
    model.eval()

    gen_preds = []
    true_labels = []

    dataloader = DataLoader(ds["test"], batch_size=16, collate_fn=data_collator)

    for batch in tqdm(dataloader, desc="Generating outputs"):
        input_ids = batch["input_ids"].cuda()
        attention_mask = (input_ids != tokenizer.pad_token_id).long().cuda()
        labels = batch["labels"]
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=inference_args["max_new_tokens"],
                num_beams=inference_args["num_beams"],
                no_repeat_ngram_size=inference_args["no_repeat_ngram_size"],
                length_penalty=inference_args["length_penalty"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_preds.append(outputs.cpu())
        true_labels.append(labels)

    # Stack tensors
    gen_preds = torch.cat(gen_preds, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    # Compute and save metrics
    metrics = compute_metrics((gen_preds, true_labels), tokenizer)

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
        "num_beams": inference_args["num_beams"],
        "no_repeat_ngram_size": inference_args["no_repeat_ngram_size"],
        "length_penalty": inference_args["length_penalty"],
        "max_new_tokens": inference_args["max_new_tokens"],
        "num_svalues_to_adapt": peft_config.num_svalues_to_adapt,
        "num_svectors_to_adapt": peft_config.num_svectors_to_adapt,
        "uiortholora_alpha": peft_config.uiortholora_alpha,
        "uiortholora_dropout": peft_config.uiortholora_dropout,
        "initial_scaler": peft_config.initial_scaler,
        "initial_sigma": peft_config.initial_sigma,
        "target_modules": list(peft_config.target_modules) if peft_config.target_modules is not None else None,  # Convert to list if it's a set
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
        if isinstance(v, (int, float)): 
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


def train_and_evaluate(model_path="outputs/models", model_type="gpt2-medium", training_args=None, finetune=False, peft_config=None, inference_args=None):
    print("training and evaluating \n", flush=True)

    # set seed and device
    seed=42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device, flush=True)   

    tokenizer = get_tokenizer(model_type)
    print("tokenizer loaded \n", flush=True)

    # load dataset
    ds = load_and_prepare(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print("dataset loaded \n", flush=True)

    if finetune:
        model = finetune_model(tokenizer, training_args, peft_config, ds, device, model_path, model_type)

    else:
        tokenizer, model, peft_config = get_tokenizer_and_model(model_path, device)
        print("Loaded already finetuned model \n", flush=True)

    # evaluate model
    evaluate_model(model, tokenizer, ds, data_collator, peft_config, training_args, inference_args)
