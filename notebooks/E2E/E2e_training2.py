import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import subprocess
from tqdm import tqdm
import numpy as np
from peft import UIOrthoLoRAConfig, get_peft_model, TaskType, PeftConfig, PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorWithPadding
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import DataLoader
import random
import transformers

SYSTEM_OUTPUTS_PATH = "system_outputs"

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU (even if you only use one)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transformers.set_seed(seed)  # affects Hugging Face Trainer, etc.

    print(f"✅ Seed set to {seed} for reproducibility")

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

        text       = prompt + reference + tokenizer.eos_token
        tok        = tokenizer(text,
                            truncation=True,
                            padding="max_length",
                            max_length=512)

        labels = tok["input_ids"].copy()
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)  # mask prompt
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]  # mask padding

        tok["labels"]     = labels
        tok["prompt_ids"] = prompt_ids                        # ✅ real prompt
        return tok

    return ds.map(to_features, remove_columns=ds["train"].column_names)


def run_evaluation(results_output_dir, run_tag):
    result = subprocess.run(
        [
            "python", "e2e-metrics/measure_scores.py",
            "--python",
            f"{results_output_dir}/testset.txt",
            f"{results_output_dir}/{run_tag}/system_outputs_uniq.txt"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Evaluation failed!")
        print("stderr:")
        print(result.stderr)
        print("stdout:")
        print(result.stdout)

    else:
        print(result.stdout)
        with open(f"{results_output_dir}/{run_tag}/scores.txt", "w", encoding="utf8") as f:
            f.write(result.stdout)

        print(f"✅ Scores written to {results_output_dir}/{run_tag}/scores.txt")


def prepare_for_evaluation(original_ds, results_output_dir, run_tag):
    # 2. Read original system outputs
    with open(f"{results_output_dir}/{run_tag}/{SYSTEM_OUTPUTS_PATH}.txt", "r", encoding="utf8") as fin:
        outputs = [line.strip() for line in fin]

    # 3. Deduplicate: keep first output per unique MR
    seen = set()
    uniq_outputs = []
    for mr, out in zip(original_ds["test"]["meaning_representation"], outputs):
        if mr not in seen:
            uniq_outputs.append(out)
            seen.add(mr)

    output_path = f"{results_output_dir}/{run_tag}/{SYSTEM_OUTPUTS_PATH}_uniq.txt"
    os.makedirs(os.path.dirname(f"{results_output_dir}/{run_tag}"), exist_ok=True)

    # 4. Write the reduced file
    with open(output_path, "w", encoding="utf8") as fout:
        for line in uniq_outputs:
            fout.write(line + "\n")

    print(f"✅ Wrote {SYSTEM_OUTPUTS_PATH}_uniq with {len(uniq_outputs)} lines")


def set_tokenizer(tokenizer, padding_side):
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        print("Added new pad token:", tokenizer.pad_token)

def set_contiguous(model):
    for m in model.modules():
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            base = m.parametrizations.weight[0].base
            if not base.is_contiguous():
                base.data = base.data.contiguous()


def get_base_model(model_type, device, tokenizer):
    base_model = AutoModelForCausalLM.from_pretrained(model_type)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model = base_model.to(device)    
    return base_model


def get_tokenizer_and_model(model_path: str, device):
    """
    Load a base model and inject a saved PEFT adapter from `model_path`.
    """
    # 1) Load the adapter config to get the original base model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    # 2) Load base model and tokenizer
    tokenizer = get_tokenizer(base_model_name)

    base_model = get_base_model(base_model_name, device, tokenizer)

    # 3) Load the adapter into the base model
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)

    # 4) Ensure contiguous weights (optional)
    set_contiguous(model)

    return tokenizer, model, peft_config



def get_tokenizer(model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    set_tokenizer(tokenizer, padding_side="right")
    return tokenizer


def finetune_model(tokenizer,training_args, orthoLoRA_args, ds, device, data_collator, model_path="outputs/models_dropout", model_type="gpt2-medium"):
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

    # lora_cfg = LoraConfig(
    # r=4,                        # rank
    # lora_alpha=16,              # α so that α / r = 4
    # lora_dropout=0.05,          # small regulariser
    # task_type=TaskType.CAUSAL_LM,
    # fan_in_fan_out=True,        # GPT-2 matrices are (out, in)
    # target_modules=[            # touch Wq & Wv only
    #     "attn.c_attn",          # covers q,k,v in one tensor
    # ],
    # )

    base_model = get_base_model(model_type, device, tokenizer)
    print("base model loaded \n", flush=True)

    # orthoLora_model = get_peft_model(base_model, orthoLoRAConfig)
    orthoLora_model = get_peft_model(base_model, orthoLoRAConfig)
    set_contiguous(orthoLora_model)
    orthoLora_model = orthoLora_model.to(device)
    orthoLora_model.print_trainable_parameters()
    print("orthoLora model loaded \n", flush=True)
    
    trainer = Trainer(
        model=orthoLora_model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator
        )

    trainer.train()
    trainer.save_model(model_path)
    print("model saved to ", model_path, flush=True)

    return trainer.model


def run_inference(
    model,
    tokenizer,
    ds,
    inference_args,
    out_dir,          # folder where we write system_outputs.txt
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
    out_path = out_dir / f"{SYSTEM_OUTPUTS_PATH}.txt"

    gen_texts = []
    tokenizer.padding_side = "left"

    def collate_fn(batch):
        feats = [{"input_ids": b["prompt_ids"]} for b in batch]
        out = tokenizer.pad(
        feats,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt"
    )

        return out        

    dataloader = DataLoader(
        ds["test"],
        batch_size=inference_args["inference_batch_size"],
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



def train_and_evaluate(output_dir, models_dir, results_dir,
                       model_type="gpt2-medium", training_args=None, finetune=False, peft_config=None,
                       inference_args=None, run_tag=None, inference=False, evaluate=False, seed=42):
    print("training and evaluating \n", flush=True)

    # set seed and device
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"{output_dir}/{models_dir}/{run_tag}"
    results_output_dir = f"{output_dir}/{results_dir}"
    print("device: ", device, flush=True)   

    tokenizer = get_tokenizer(model_type)
    print("tokenizer loaded \n", flush=True)

    # load dataset
    ds = load_and_prepare(tokenizer)
    original_ds = load_dataset("tuetschek/e2e_nlg", trust_remote_code=True)
    print("dataset loaded \n", flush=True)

    if finetune:
        data_collator = DataCollatorWithPadding(tokenizer, padding=True)
        model = finetune_model(tokenizer, training_args, peft_config, ds, device, data_collator, model_path, model_type)

    else:
        tokenizer, model, peft_config = get_tokenizer_and_model(model_path, device)
        print("Loaded already finetuned model \n", flush=True)

    # run inference and save results
    if inference:
        print("setting tokenizer padding side to left for inference \n", flush=True)
        set_tokenizer(tokenizer, padding_side="left")
        run_inference(model, tokenizer, ds, inference_args, out_dir=training_args.output_dir)

    if evaluate:
        prepare_for_evaluation(original_ds, results_output_dir, run_tag)
        run_evaluation(results_output_dir, run_tag)
