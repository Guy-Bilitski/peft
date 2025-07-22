# training_manager.py  – hyper-parameter search over learning-rate
from pathlib import Path
from transformers.training_args import TrainingArguments
from E2e_training2 import train_and_evaluate
from peft import UIOrthoLoRAConfig
import itertools

# ───────────────────────── settings every run shares ───────────────────────── #
MODEL_TYPE = "gpt2-medium"
FINETUNE   = True   # change to False if you only want to evaluate
INFERENCE = True
EVALUATE = True
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
RESULTS_DIR = "results"

LORA_CFG = UIOrthoLoRAConfig(
    target_modules = [
        "attn.c_attn", "attn.c_proj",
    ],
    fan_in_fan_out=True,
    initial_scaler=0.1,
    initial_sigma=0.1,
    uiortholora_alpha=1,
    uiortholora_dropout=0,
    num_svalues_to_adapt=128,
    num_svectors_to_adapt=45,
)

BASE_TRAIN_ARGS = dict(
    overwrite_output_dir=True,
    eval_strategy="no",
    save_strategy="no",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_accumulation_steps=2,
    lr_scheduler_type="linear",
    label_smoothing_factor=0.1,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=50,
    save_total_limit=1,
    report_to="none",
)

INFERENCE_ARGS = {
    "num_beams": 10,
    "no_repeat_ngram_size": 4,
    "length_penalty": 0.9,
    "max_new_tokens": 64,
    "inference_batch_size": 16,
}

# ───────────── grid of learning-rates to try ───────────── #
SEARCH_LRS = [7e-2, 8e-2, 9e-2, 1e-1]
NUM_SVALUES = [256]
NUM_SVECTORS = [60]
# SEEDS = [2021, 17, 31415, 1054]
SEEDS = [31415]
INIT_SIGMA = [0.01]
INIT_SCALER = [0.01]

# ───────────────────────── main loop ───────────────────── #
def main() -> None:
    for lr, num_svalues, num_svectors, seed, init_sigma, init_scaler in itertools.product(SEARCH_LRS, NUM_SVALUES, NUM_SVECTORS, SEEDS, INIT_SIGMA, INIT_SCALER):
        # unique sub-folders per LR
        run_tag      = f"lr_{lr:g}_svalues_{num_svalues}_svectors_{num_svectors}_seed_{seed}_init_sigma_{LORA_CFG.initial_sigma}_init_scaler_{LORA_CFG.initial_scaler}"
        model_path   = f"{OUTPUT_DIR}/{MODELS_DIR}/{run_tag}"
        results_path = f"{OUTPUT_DIR}/{RESULTS_DIR}/{run_tag}"

        LORA_CFG.num_svalues_to_adapt = num_svalues
        LORA_CFG.num_svectors_to_adapt = num_svectors
        LORA_CFG.initial_sigma = init_sigma
        LORA_CFG.initial_scaler = init_scaler

        Path(model_path).mkdir(parents=True, exist_ok=True)
        Path(results_path).mkdir(parents=True, exist_ok=True)

        train_args = TrainingArguments(
            output_dir=results_path,
            learning_rate=lr,
            **BASE_TRAIN_ARGS,
        )

        print(f"\n▶️  Starting run {run_tag}")
        train_and_evaluate(
            output_dir=OUTPUT_DIR,
            models_dir=MODELS_DIR,
            results_dir=RESULTS_DIR,
            model_type=MODEL_TYPE,
            training_args=train_args,
            finetune=FINETUNE,
            peft_config=LORA_CFG,
            inference_args=INFERENCE_ARGS,
            run_tag=run_tag,
            inference=INFERENCE,
            evaluate=EVALUATE,
            seed=seed
        )
        print(f"✅ Finished run {run_tag}")

if __name__ == "__main__":
    # ensure the root folders exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/{MODELS_DIR}").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/{RESULTS_DIR}").mkdir(parents=True, exist_ok=True)
    main()
