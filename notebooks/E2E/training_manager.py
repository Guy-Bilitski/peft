
# training_manager.py
"""Minimal launcher for e2e_full_training.train_and_evaluate.
Change the variables below (model_path, finetune, lora_cfg fields) to run
with different configurations; no command‑line arguments are parsed.
"""

from pathlib import Path
from transformers.training_args import TrainingArguments

from e2e_full_training import (
    train_and_evaluate,
    UIOrthoLoRAConfig,
)

# ─────────────────────────── USER‑EDITABLE SECTION ─────────────────────────────
MODEL_PATH = "outputs/models"   # directory to save / load the PEFT model
FINETUNE   = True               # True → run fine‑tuning before evaluation
MODEL_TYPE = "gpt2-medium"     # model type to use

LORA_CFG = UIOrthoLoRAConfig(
    target_modules=["attn.c_attn", "attn.c_proj"],
    fan_in_fan_out=True,
    initial_scaler=0.1,
    initial_sigma=0.1,
    uiortholora_alpha=1,
    uiortholora_dropout=0,
    num_svalues_to_adapt=96,
    num_svectors_to_adapt=30,
    task_type=None,  # will be set inside train_and_evaluate
)

TRAINING_ARGS = TrainingArguments(
    output_dir="outputs/results",
    overwrite_output_dir=True,
    eval_strategy="no",
    save_strategy="no",
    per_device_train_batch_size=8,
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
    report_to="none")

INFERENCE_ARGS = {
    "num_beams": 10,
    "no_repeat_ngram_size": 4,
    "length_penalty": 0.9,
    "max_new_tokens": 64,
}

# ───────────────────────────────────────────────────────────────────────────────

def main() -> None:
    train_and_evaluate(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        training_args=TRAINING_ARGS,
        finetune=FINETUNE,
        peft_config=LORA_CFG,
        inference_args=INFERENCE_ARGS,
    )


if __name__ == "__main__":
    # Create output directory if it doesn’t exist
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/results").mkdir(exist_ok=True)
    main()
