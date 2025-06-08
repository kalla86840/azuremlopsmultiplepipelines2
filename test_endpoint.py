import requests
import json

# Replace with your actual scoring URI after deployment
scoring_uri = "http://<your-endpoint-id>.eastus.azurecontainer.io/score"

# Example input for cars model (adjust based on model used)
sample_data = {
    "data": [[2.0, 150, 3000, 2020]]  # [engine_size, horsepower, weight, year]
}

headers = {"Content-Type": "application/json"}

try:
    response = requests.post(scoring_uri, headers=headers, json=sample_data)
    print("Prediction:", response.json())
except Exception as e:
    print("Request failed:", str(e))