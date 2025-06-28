#!/usr/bin/env python
"""
Fine-tune Mistral-7B (or any HF causal-LM) on GSM8K with UIOrthoLoRA.

Usage (single-GPU example):
python train_gsm8k_uiortholora.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --output_dir runs/mistral_uiolora \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --bits 8

Resume / load existing adapters:
    --adapter_path runs/mistral_uiolora/adapter_model
"""

import logging, os, random, datetime, json, torch, numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Dict

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import (
    UIOrthoLoRAConfig, get_peft_model, PeftModel, TaskType,
    prepare_model_for_kbit_training,
)

# ────────────────────────── logging ────────────────────────── #
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
# ───────────────────────────────────────────────────────────── #

PROMPT_TMPL = "Q: {question}\nA:"          # identical to eval script
IGNORE_IDX  = -100


# ════════════════  CLI / dataclass  ════════════════
@dataclass
@dataclass
class ScriptArgs(TrainingArguments):
    # --- model ---------------------------------------------------------------
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "HF model to fine-tune."},
    )

    # allow resume
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a saved UIOrthoLoRA adapter (resume training)."},
    )

    # --- UIOrthoLoRA hyper-params -------------------------------------------
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated Linear layer names to adapt."},
    )
    fan_in_fan_out: bool = field(default=False)
    uiortholora_alpha: float = field(default=32.0)
    uiortholora_dropout: float = field(default=0.05)
    num_svalues_to_adapt: int = field(default=8)
    num_svectors_to_adapt: int = field(default=8)
    initial_scaler: float = field(default=0.1)
    initial_sigma: float = field(default=0.02)

    # data / misc
    max_prompt_len: int = field(default=512)
    seed: int = field(default=42)


# ════════════════  data helpers  ════════════════ #
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _prompt_tokenise(example, tok, max_len):
    """Build 'Q: … \\nA:' and mask question tokens in labels."""
    txt = PROMPT_TMPL.format(question=example["question"])
    enc = tok(txt, truncation=True, max_length=max_len)
    enc["labels"] = enc["input_ids"].copy()
    # mask prompt tokens
    enc["labels"][: len(enc["input_ids"])] = [IGNORE_IDX] * len(enc["input_ids"])
    return enc


# ════════════════  checkpoint utils  ════════════════ #
def _last_ckpt(out_dir: str):
    if not os.path.isdir(out_dir): return None
    if os.path.exists(os.path.join(out_dir, "completed")):   # finished run
        return None
    cks = [d for d in os.listdir(out_dir) if d.startswith(PREFIX_CHECKPOINT_DIR)]
    if not cks: return None
    step = max(int(c.split("-")[-1]) for c in cks)
    ckpt = os.path.join(out_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}")
    logger.info("Resuming from checkpoint %s", ckpt)
    return ckpt


class SaveAdapterCallback(transformers.TrainerCallback):
    """Saves only PEFT adapter during checkpoints."""
    def __init__(self, tok): self.tok = tok

    def _save(self, args, state, model):
        step_dir = state.best_model_checkpoint or os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        ad_path = os.path.join(step_dir, "adapter_model")
        model.save_pretrained(ad_path)
        self.tok.save_pretrained(ad_path)

    def on_save(self, args, state, control, **kwargs):
        self._save(args, state, kwargs["model"]); return control
    def on_train_end(self, args, state, control, **kwargs):
        Path(args.output_dir, "completed").touch()
        self._save(args, state, kwargs["model"])


# ════════════════  build model  ════════════════
def build_model(args: ScriptArgs, tok):
    """
    Load the base model and attach (or reload) a UIOrthoLoRA adapter.
    """
    # 1️⃣  Base model ----------------------------------------------------------
    compute_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    # 2️⃣  Adapter: resume or fresh -------------------------------------------
    if args.adapter_path:
        logger.info("🔄  Loading existing UIOrthoLoRA adapter from %s", args.adapter_path)
        model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)

    else:
        logger.info("✨  Initialising NEW UIOrthoLoRA adapter")
        cfg = UIOrthoLoRAConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=args.target_modules.split(","),
            fan_in_fan_out=args.fan_in_fan_out,
            initial_scaler=args.initial_scaler,
            initial_sigma=args.initial_sigma,
            uiortholora_alpha=args.uiortholora_alpha,
            uiortholora_dropout=args.uiortholora_dropout,
            num_svalues_to_adapt=args.num_svalues_to_adapt,
            num_svectors_to_adapt=args.num_svectors_to_adapt,
        )
        model = get_peft_model(base, cfg)

    return model



# ════════════════  main train()  ════════════════ #
def train():
    parser = transformers.HfArgumentParser(ScriptArgs)
    args: ScriptArgs = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed); logger.info("args=%s", args)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    # dataset
    ds_train = load_dataset("gsm8k", "main", split="train").map(
        lambda ex: _prompt_tokenise(ex, tok, args.max_prompt_len),
        remove_columns=["question", "answer"],
        batched=False,
    )

    model = build_model(args, tok)
    logger.info(model)

    # HF TrainingArguments already in `args` (ScriptArgs inherits)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        data_collator=lambda x: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(i["input_ids"]) for i in x],
                batch_first=True,
                padding_value=tok.pad_token_id,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(i["labels"]) for i in x],
                batch_first=True,
                padding_value=IGNORE_IDX,
            ),
            "attention_mask": None,
        },
        tokenizer=tok,
    )
    trainer.add_callback(SaveAdapterCallback(tok))

    resume_ckpt = _last_ckpt(args.output_dir)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_state()

    # final save: merged FP16 for eval script
    merged_dir = Path(args.output_dir, "merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.merge_and_unload().save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)

    # log metrics
    final_metrics = {
        "end_time": datetime.datetime.now().isoformat(),
        "train_samples": len(ds_train),
        **{k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
    }
    Path(args.output_dir, "train_args_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    logger.info("Training complete; merged model saved to %s", merged_dir)


if __name__ == "__main__":
    train()
