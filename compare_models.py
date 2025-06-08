import os
import json
import argparse
from datetime import datetime

os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)

def compare_models(dataset):
    models = ["LinearRegression", "DecisionTree", "RandomForest"]
    best_model = None
    best_r2 = float('-inf')
    best_rmse = float('inf')
    metrics_summary = {}

    for model in models:
        metric_path = f"outputs/{dataset}_{model}_metrics.json"
        if not os.path.exists(metric_path):
            print(f"Metrics not found for {model} at {metric_path}. Skipping.")
            continue

        metrics = load_metrics(metric_path)
        r2 = metrics.get("r2_score", None)
        rmse = metrics.get("rmse", None)

        if r2 is None or rmse is None:
            print(f"Missing r2_score or rmse in {metric_path}. Skipping.")
            continue

        metrics_summary[model] = {"r2_score": r2, "rmse": rmse}

        if (r2 > best_r2) or (r2 == best_r2 and rmse < best_rmse):
            best_r2 = r2
            best_rmse = rmse
            best_model = model

    if not best_model:
        raise ValueError("No valid model metrics found for comparison.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/metrics_report_{dataset}_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump({
            "best_model": best_model,
            "models": metrics_summary,
            "timestamp": timestamp
        }, f, indent=2)

    with open(f"outputs/{dataset}_best_model.txt", "w") as f:
        f.write(best_model)

    print(f"Best model: {best_model}")
    print(f"Metrics report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (bikes, cars, house)")
    args = parser.parse_args()
    compare_models(args.dataset)
