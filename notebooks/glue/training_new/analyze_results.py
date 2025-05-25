from clearml import Task
import pandas as pd
from tqdm import tqdm

def main():
    project_name = "GLUE benchmark"
    task_filter = "rte"

    print("Fetching tasks...")
    try:
        tasks = Task.get_tasks(project_name=project_name)
    except Exception as e:
        print("Error fetching tasks:", e)
        return

    print(f"Found {len(tasks)} tasks")

    records = []
    for task in tqdm(tasks, desc="Processing tasks"):
        if task_filter in task.name.lower() and task.status == "completed":
            params = task.get_parameters()
            scalars = task.get_last_scalar_metrics()

            score = None
            for title, metrics in scalars.items():
                for metric, series in metrics.items():
                    if "eval_accuracy" in metric or "score" in metric:
                        score = series.get("value", None)

            records.append({
                "task_name": task.name,
                "rank": params.get("rank"),
                "head_lr": params.get("head_lr"),
                "adapter_lr": params.get("adapter_lr"),
                "initial_scaler": params.get("initial_scaler"),
                "initial_sigma": params.get("initial_sigma"),
                "score": score,
                "id": task.id
            })

    df = pd.DataFrame(records)
    df.to_excel("clearml_tuning_summary.xlsx", index=False)
    print("Saved to clearml_tuning_summary.xlsx")

if __name__ == "__main__":
    main()
