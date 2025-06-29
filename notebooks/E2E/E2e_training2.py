import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from tqdm import tqdm
import evaluate
import numpy as np
from peft import UIOrthoLoRAConfig, get_peft_model, TaskType, PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorWithPadding
from datasets import load_dataset
from pathlib import Path
import json
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader
import datetime
from transformers.data.data_collator import default_data_collator

def load_and_prepare(tokenizer):
    ds = load_dataset("tuetschek/e2e_nlg", trust_remote_code=True)

    def to_features(rec):
        prompt    = f"{rec['meaning_representation']} => "
        reference = rec.get("human_reference") or rec.get("reference", "")

        # 1️⃣  tokenize prompt **alone** (no padding, no special tokens)
        prompt_ids = tokenizer(prompt,
                            add_special_tokens=False,
                            padding=False,
                            truncation=False)["input_ids"]

        # 2️⃣  tokenize prompt + reference with left-padding to 512
        text       = prompt + reference
        tok        = tokenizer(text,
                            truncation=True,
                            padding="max_length",
                            max_length=512)

        labels = tok["input_ids"].copy()
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)   # mask prompt

        tok["labels"]     = labels
        tok["prompt_ids"] = prompt_ids                        # ✅ real prompt
        return tok

    return ds.map(to_features, remove_columns=ds["train"].column_names)


def set_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def set_contiguous(model):
    for m in model.modules():
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            base = m.parametrizations.weight[0].base
            if not base.is_contiguous():
                base.data = base.data.contiguous()


def get_tokenizer_and_model(model_path: str, device):
    """
    Load a base model and inject a saved PEFT adapter from `model_path`.
    """
    # 1) Load the adapter config to get the original base model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    # 2) Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    set_tokenizer(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    base_model = base_model.to(device)

    # 3) Load the adapter into the base model
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)

    # 4) Ensure contiguous weights (optional)
    set_contiguous(model)

    return tokenizer, model, peft_config



def get_tokenizer(model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False)
    set_tokenizer(tokenizer)

    return tokenizer


def finetune_model(tokenizer,training_args, orthoLoRA_args, ds, device, data_collator, model_path="outputs/models", model_type="gpt2-medium"):
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
        train_dataset=ds["train"].select(range(1000)), # TODO: remove the select 
        eval_dataset=ds["validation"].select(range(100)),
        data_collator=data_collator
        )

    trainer.train()
    trainer.save_model(model_path)
    print("model saved to ", model_path, flush=True)

    return trainer.model


def evaluate_model(
    model,
    tokenizer,
    ds,
    inference_args,
    out_dir="outputs",          # folder where we write system_outputs.txt
):
    """
    Generate E2E test predictions and save them to
    <out_dir>/system_outputs.txt  – one sentence per line, same order as ds["test"].
    Metrics are NOT computed here; run e2e-metrics afterwards.
    """
    model.eval()
    device = model.device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "system_outputs.txt"

    gen_texts = []

    def collate_fn(batch):
        feats = [{"input_ids": b["prompt_ids"]} for b in batch]
        out = tokenizer.pad(
        feats,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt"
    )


        for idx, example in enumerate(batch):
            print("\n🔹 Original prompt:", tokenizer.decode(example["prompt_ids"],
                                                        skip_special_tokens=False))
            padded_ids = out["input_ids"][idx].tolist()
            print("🔹 Padded tokens:  ", tokenizer.convert_ids_to_tokens(padded_ids))

        return out        

    dataloader = DataLoader(
        ds["test"].select(range(100)),        # TODO: remove the select
        batch_size=32,
        collate_fn=collate_fn,
    )

    for item in tqdm(dataloader):
        prompt_ids = item["input_ids"].to(device)
        attention_mask = item["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=inference_args["max_new_tokens"],
                num_beams=inference_args["num_beams"],
                no_repeat_ngram_size=inference_args["no_repeat_ngram_size"],
                length_penalty=inference_args["length_penalty"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # strip the prompt text (“=>” part) to isolate hypothesis
        preds = [p.split("=>")[-1].strip() for p in preds]
        gen_texts.extend(preds)


    # ── save to file ────────────────────────────────────────────────
    with out_path.open("w", encoding="utf8") as f:
        for line in gen_texts:
            f.write(line + "\n")

    print(f"📝 Saved {len(gen_texts)} predictions → {out_path}")
    print("Now run the official scorer, e.g.:")
    print("docker run --rm -v $(pwd)/outputs:/data -v PATH/TO/e2e-metrics/references:/refs "
          "e2e-metrics ./measure_scores.py -p /data/system_outputs.txt -r /refs -s")



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
    print("dataset loaded \n", flush=True)

    if finetune:
        data_collator = DataCollatorWithPadding(tokenizer, padding=True)
        model = finetune_model(tokenizer, training_args, peft_config, ds, device, data_collator, model_path, model_type)

    else:
        tokenizer, model, peft_config = get_tokenizer_and_model(model_path, device)
        tokenizer.padding_side = "left"
        print("Loaded already finetuned model \n", flush=True)

    # evaluate model
    evaluate_model(model, tokenizer, ds, inference_args, out_dir=training_args.output_dir)
