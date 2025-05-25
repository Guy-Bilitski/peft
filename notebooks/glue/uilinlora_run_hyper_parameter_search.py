import os
import sys
import subprocess
from itertools import product

if len(sys.argv) < 2:
    print("Usage: python run_hyperparam_search.py <task>")
    sys.exit(1)

task_name = sys.argv[1]
gpu_id = "0"

# Define search space (SST-2)
ranks = [128, 256]
head_lrs = [4e-4, 1e-3, 4e-3]
adapter_lrs = [4e-4, 1e-3, 4e-3]
initial_scalers = [1e-7, 1e-2, 1e-1, 1]
initial_sigmas  = [1e-7, 1e-2, 1e-1, 1]

search_space = list(product(ranks, head_lrs, adapter_lrs, initial_scalers, initial_sigmas))

for i, (rank, head_lr, adapter_lr, scaler, sigma) in enumerate(search_space):
    print(f"\n=== Run {i+1}/{len(search_space)} ===")
    cmd = [
        "python", "uilinlora_robertabase_training.py",
        "--task", task_name,
        "--rank", str(rank),
        "--head_lr", str(head_lr),
        "--adapter_lr", str(adapter_lr),
        "--initial_scaler", str(scaler),
        "--initial_sigma", str(sigma),
        "--batch_size", "64",
        "--epochs", "15"  # Or adjust based on task
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    subprocess.run(cmd, env=env)
