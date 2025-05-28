import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from training import train_model
import argparse
from itertools import product

# Config
task_name = "rte"
epochs = 30
batch_size = 64
max_len = 256
base_model_id = "roberta-base"
model_type = "uiortholora"
uiortholora_alpha = 1
uiortholora_dropout = 0
target_modules = ["attention.output.dense", "query", "key", "value"]

# Define search space
num_svalues_to_adapt = [128]
num_svectors_to_adapt = [60]
head_lrs = [5e-4, 1e-3, 5e-3, 1e-2]
adapter_lrs = [1e-3, 5e-3, 1e-2, 4e-2]
initial_scalers = [1e-1, 1e-2]
initial_sigmas  = [1e-1, 1e-2]

# Cartesian product of all configs
search_space = list(product(
    num_svalues_to_adapt, num_svectors_to_adapt,
    head_lrs, adapter_lrs, initial_scalers, initial_sigmas
))

for i, (num_svalues, num_svectors, head_lr, adapter_lr, scaler, sigma) in enumerate(search_space):
    print(f"\n=== Run {i+1}/{len(search_space)} ===")

    args = argparse.Namespace(
        task=task_name,
        epochs=epochs,
        seed=42,
        num_svalues_to_adapt=num_svalues,
        num_svectors_to_adapt=num_svectors,
        head_lr=head_lr,
        adapter_lr=adapter_lr,
        initial_scaler=scaler,
        initial_sigma=sigma,
        batch_size=batch_size,
        max_len=max_len,
        base_model_id=base_model_id,
        model_type=model_type,
        uiortholora_alpha=uiortholora_alpha,
        uiortholora_dropout=uiortholora_dropout,
        target_modules=target_modules
    )

    train_model(args)
