import os
import argparse
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    models = ["LinearRegression", "DecisionTree", "RandomForest"]
    metrics_report = {}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))  # Go two levels up

    for model_name in models:
        y_true_path = os.path.join(root_dir, f"y_true_{args.dataset}_{model_name}.txt")
        y_pred_path = os.path.join(root_dir, f"y_pred_{args.dataset}_{model_name}.txt")

        if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
            print(f"Missing prediction or ground truth files for {model_name} on {args.dataset}")
            continue

        with open(y_true_path, "r") as f:
            y_true = [float(line.strip()) for line in f]

        with open(y_pred_path, "r") as f:
            y_pred = [float(line.strip()) for line in f]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics_report[model_name] = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        }

    os.makedirs(os.path.join(root_dir, "reports"), exist_ok=True)
    out_path = os.path.join(root_dir, "reports", f"metrics_report_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"Metrics report saved to: {out_path}")

if __name__ == "__main__":
    main()