import os
import sys
import subprocess
from itertools import product

if len(sys.argv) < 2:
    print("Usage: python run_hyperparam_search.py <task>")
    sys.exit(1)

task_name = sys.argv[1]
gpu_id = "1"

# Define search space (SST-2)
num_svalues_to_adapt = [128, 256]
num_svectors_to_adapt = [40, 80]
head_lrs = [5e-3]
adapter_lrs = [5e-3]
initial_scalers = [1e-1]
initial_sigmas  = [1e-1]

search_space = list(product(num_svalues_to_adapt, num_svectors_to_adapt, head_lrs, adapter_lrs, initial_scalers, initial_sigmas))

for i, (num_svalues_to_adapt, num_svectors_to_adapt, head_lr, adapter_lr, initial_scaler, initial_sigma) in enumerate(search_space):
    print(f"\n=== Run {i+1}/{len(search_space)} ===")
    cmd = [
        "python", "uiortholora_training_new.py",
        "--task", task_name,
        "--num_svalues_to_adapt", str(num_svalues_to_adapt),
        "--num_svectors_to_adapt", str(num_svectors_to_adapt),
        "--head_lr", str(head_lr),
        "--adapter_lr", str(adapter_lr),
        "--initial_scaler", str(initial_scaler),
        "--initial_sigma", str(initial_sigma),
        "--batch_size", "64",
        "--epochs", "160"
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    subprocess.run(cmd, env=env)
