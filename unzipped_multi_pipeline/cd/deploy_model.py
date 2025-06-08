import os
import sys
import joblib
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice

def deploy_model(dataset, config_path="config.json"):
    best_model_file = f"outputs/{dataset}_best_model.txt"
    if not os.path.exists(best_model_file):
        raise FileNotFoundError(f"Missing: {best_model_file}")

    with open(best_model_file, "r") as f:
        model_type = f.read().strip()

    model_path = f"models/{model_type.lower()}/model.joblib"
    model_name = f"{dataset}_{model_type}_model"

    ws = Workspace.from_config(path=config_path)

    # Register the model
    registered_model = Model.register(
        model_path=model_path,
        model_name=model_name,
        workspace=ws
    )
    print(f"Model registered: {registered_model.name}")

    # Define inference config
    env = Environment.from_conda_specification(name=f"{model_type}-env", file_path="environment/environment.yml")
    inference_config = InferenceConfig(entry_script=f"models/{model_type.lower()}/score.py", environment=env)

    # Define deployment config
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

    service_name = f"{dataset}-{model_type.lower()}-svc".replace("_", "-")
    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=[registered_model],
        inference_config=inference_config,
        deployment_config=aci_config,
        overwrite=True
    )

    service.wait_for_deployment(show_output=True)
    print(f"Scoring URI: {service.scoring_uri}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python deploy_model.py <dataset>")
        sys.exit(1)
    deploy_model(sys.argv[1])