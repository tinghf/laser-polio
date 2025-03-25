import os
import yaml
from kubernetes import client, config

# Load Kubernetes config
config.load_kube_config()

# Kubernetes API client
batch_v1 = client.BatchV1Api()

# Load and process the manifest YAML file
yaml_file = "laser-study-creator-deploy-manifests.yaml"

with open(yaml_file, "r") as f:
    manifest = yaml.safe_load(f)

# Replace environment variables in the manifest
env_vars = {
    "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD", "default_password"),
    "MYSQL_USER": os.getenv("MYSQL_USER", "default_user"),
    "MYSQL_DB": os.getenv("MYSQL_DB", "default_db"),
    "MYSQL_ROOT_PASSWORD": os.getenv("MYSQL_ROOT_PASSWORD", "root_password"),
    "STUDY_NAME": os.getenv("STUDY_NAME", "default_study"),
}

# Function to recursively replace env variables in the YAML structure
def replace_env_vars(obj):
    if isinstance(obj, dict):
        return {k: replace_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_env_vars(i) for i in obj]
    elif isinstance(obj, str):
        for key, val in env_vars.items():
            obj = obj.replace(f"${{{key}}}", val)
        return obj
    return obj

manifest = replace_env_vars(manifest)

# Apply the job definition
namespace = "default"  # Change if using a different namespace

try:
    response = batch_v1.create_namespaced_job(namespace=namespace, body=manifest)
    print(f"Job {response.metadata.name} created successfully.")
except client.exceptions.ApiException as e:
    print(f"Error applying the job: {e}")