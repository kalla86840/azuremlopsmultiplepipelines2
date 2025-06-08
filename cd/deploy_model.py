import sys
import os
import json
from azureml.core import Workspace, Model, Environment, Webservice
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

if len(sys.argv) != 2:
    print("Usage: python deploy_model.py <dataset>")
    sys.exit(1)

dataset = sys.argv[1]
best_model_file = f"outputs/{dataset}_best_model.txt"

if not os.path.exists(best_model_file):
    print(f"[ERROR] Best model file not found: {best_model_file}")
    sys.exit(1)

with open(best_model_file, "r") as f:
    best_model_name = f.read().strip()

print(f"[INFO] Best model to deploy for {dataset}: {best_model_name}")

# Load Azure ML workspace
with open(f"config/{dataset}_config.json") as f:
    config = json.load(f)

ws = Workspace(
    subscription_id=config["subscription_id"],
    resource_group=config["resource_group"],
    workspace_name=config["workspace_name"]
)

# Map model name to subfolder
folder_map = {
    "LinearRegression": "linear_regression",
    "DecisionTree": "decision_tree",
    "RandomForest": "random_forest"
}
model_folder = folder_map.get(best_model_name, best_model_name.lower())
model_path = f"models/{dataset}/{model_folder}/model.joblib"

if not os.path.exists(model_path):
    print(f"[ERROR] Trained model file not found at {model_path}")
    sys.exit(1)

model = Model.register(
    workspace=ws,
    model_path=model_path,
    model_name=f"{dataset}_{best_model_name}"
)

# Create environment
env = Environment(name=f"{dataset}-env")
env.python.user_managed_dependencies = False
env.docker.enabled = True
env.python.conda_dependencies_file = f"environment/requirements_{dataset}.txt"

# Setup inference config and deployment config
inference_config = InferenceConfig(entry_script="score.py", environment=env)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service_name = f"{dataset}-{best_model_name.lower()}-svc"
print(f"[INFO] Deploying model to ACI as service: {service_name}")

service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(f"[SUCCESS] Service state: {service.state}")
print(f"[SUCCESS] Scoring URI: {service.scoring_uri}")