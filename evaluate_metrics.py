import os
import json
import sys
from sklearn.metrics import r2_score, mean_squared_error

os.makedirs("outputs", exist_ok=True)

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    dataset = sys.argv[1]
    model_name = sys.argv[2]

    y_true_path = f"outputs/{dataset}_{model_name}_y_true.json"
    y_pred_path = f"outputs/{dataset}_{model_name}_y_pred.json"

    y_true = load_json(y_true_path)
    y_pred = load_json(y_pred_path)

    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    metrics = {
        "r2_score": r2,
        "rmse": rmse
    }

    metrics_path = f"outputs/{dataset}_{model_name}_metrics.json"
    save_json(metrics_path, metrics)
