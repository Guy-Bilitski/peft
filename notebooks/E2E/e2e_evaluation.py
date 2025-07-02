"""evaluation_manager.py
Run the **official E2E‑NLG metrics docker image** on every experiment
folder produced by `training_manager.py` and collect the scores in a
single CSV / JSON file for convenient comparison.

Usage
-----
.. code-block:: bash

    # evaluate every run under outputs/results/<run_tag>
    python evaluation_manager.py \
           --results-root outputs/results \
           --refs-dir   /ABS/PATH/TO/e2e-metrics/references \
           --output     outputs/metrics.csv

Notes
-----
* We assume the docker image `e2e-metrics` is already pulled/built.
  See: <https://github.com/kedz/e2e-metrics>
* Each run folder must contain a *system_outputs.txt* file as written by
  `evaluate_model()`.
* The script mounts **only the run folder** into the container, so the
  scorer always reads **/data/system_outputs.txt**.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

_METRIC_RE = re.compile(r"([A-Za-z]+)\s*=\s*([0-9.]+)")


def _parse_metrics(stdout: str) -> Dict[str, float]:
    """Extract metrics from the scorer's stdout.

    Expected lines look like::
        BLEU = 0.6715
        NIST = 8.2210
        METEOR = 0.4678
        ROUGE_L = 0.7193
        CIDEr = 2.3819
    """
    metrics = {}
    for line in stdout.splitlines():
        m = _METRIC_RE.search(line.strip())
        if m:
            metric_name, value = m.groups()
            metrics[metric_name] = float(value)
    return metrics


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def evaluate_run(run_dir: Path, refs_dir: Path) -> Dict[str, float]:
    """Run docker‑based evaluation for **one** experiment directory.

    Parameters
    ----------
    run_dir : Path
        Directory that contains *system_outputs.txt*.
    refs_dir : Path
        Path to the reference texts directory provided by the official
        e2e‑metrics repo.  Mounted read‑only inside the container.
    """
    sys_out = run_dir / "system_outputs.txt"
    if not sys_out.exists():
        raise FileNotFoundError(f"{sys_out} not found – have you run inference?")

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{run_dir.resolve()}:/data",       # mount this run
        "-v",
        f"{refs_dir.resolve()}:/refs:ro",  # mount references read‑only
        "e2e-metrics",
        "./measure_scores.py",
        "-p",
        "/data/system_outputs.txt",
        "-r",
        "/refs",
        "-s",  # silent progress bar, nicer parsing
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    metrics = _parse_metrics(proc.stdout)
    if not metrics:
        raise RuntimeError(
            f"Could not parse metrics for {run_dir}\nOutput was:\n{proc.stdout}\n{proc.stderr}"
        )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate all experiment runs with the official E2E‑NLG scorer (docker)")
    p.add_argument("--results-root", type=Path, default=Path("outputs/results"), help="Root directory that contains per‑run folders (default: outputs/results)")
    p.add_argument("--refs-dir", type=Path, required=True, help="Path to e2e‑metrics/references directory")
    p.add_argument("--output", type=Path, default=Path("outputs/metrics.csv"), help="Where to store the aggregated metrics table (CSV)")
    p.add_argument("--as-json", action="store_true", help="In addition to CSV, also dump a JSON file next to it")
    args = p.parse_args()

    run_dirs = sorted([p for p in args.results_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No runs found under {args.results_root}")

    print(f"Found {len(run_dirs)} run(s) under {args.results_root}\n")

    # Collect metrics for every run
    all_rows: List[Tuple[str, Dict[str, float]]] = []
    for run_dir in run_dirs:
        tag = run_dir.name
        print(f"▶️  Evaluating {tag} …", end=" ", flush=True)
        try:
            scores = evaluate_run(run_dir, args.refs_dir)
        except Exception as e:
            print("❌ failed")
            print(e)
            continue
        print("✅ done")
        all_rows.append((tag, scores))

    if not all_rows:
        raise SystemExit("No successful evaluations – aborting")

    # Determine union of metric names to make the header stable
    metric_names: List[str] = sorted({k for _, m in all_rows for k in m})
    out_file = args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with out_file.open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_tag", *metric_names])
        for tag, scores in all_rows:
            row = [scores.get(m, "") for m in metric_names]
            writer.writerow([tag, *row])

    print(f"\n✅ Metrics saved → {out_file.relative_to(Path.cwd())}")

    # Optional JSON dump
    if args.as_json:
        json_path = out_file.with_suffix(".json")
        with json_path.open("w", encoding="utf8") as f:
            json.dump({tag: scores for tag, scores in all_rows}, f, indent=2)
        print(f"✅ Also wrote {json_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
