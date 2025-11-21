# register model

import json
import mlflow
import logging
from src.logger import logging
import os
# import dagshub
from dotenv import load_dotenv

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"
# Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------

# mlflow.set_tracking_uri("https://dagshub.com/Jugal-lachhwani/Data-Capstone-Project.mlflow")
# dagshub.init(repo_owner='Jugal-lachhwani', repo_name='Data-Capstone-Project', mlflow=True)# -------------------------------------------------------------------------------------

load_dotenv()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/Jugal-lachhwani/Data-Capstone-Project.mlflow")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set credentials if available (for CI)
if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")
else:
    print("Authorization Required")
    # For local use with DagHub OAuth
    # dagshub.init(repo_owner='Jugal-lachhwani', repo_name='Data-Capstone-Project', mlflow=True)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        return model_version.version 
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise
    
def promote_model_to_production(model_name: str, staging_version: str, metric_name: str = "accuracy"):
    """Promote the staging model to production if better than current production."""
    client = mlflow.tracking.MlflowClient()

    # Fetch all model versions
    versions = client.search_model_versions(f"name='{model_name}'")
    
    # Separate production and staging versions
    staging_versions = [v for v in versions if v.current_stage == "Staging"]
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    
    if not staging_versions:
        logging.warning("No staging models found.")
        return

    # Get the latest staging version (the one we just registered)
    latest_staging = [v for v in staging_versions if str(v.version) == str(staging_version)][0]
    staging_run = client.get_run(latest_staging.run_id)
    staging_metric = staging_run.data.metrics.get(metric_name, None)

    if staging_metric is None:
        logging.warning(f"Metric '{metric_name}' not found in staging model run. Skipping promotion.")
        return

    # If no production model exists, promote directly
    if not prod_versions:
        logging.info(f"No production model found. Promoting staging model v{latest_staging.version} to Production.")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging.version,
            stage="Production"
        )
        return

    # Compare with latest production model
    latest_prod = sorted(prod_versions, key=lambda x: int(x.version))[-1]
    prod_run = client.get_run(latest_prod.run_id)
    prod_metric = prod_run.data.metrics.get(metric_name, None)

    logging.info(f"Staging v{latest_staging.version} {metric_name}: {staging_metric}")
    logging.info(f"Production v{latest_prod.version} {metric_name}: {prod_metric}")

    if prod_metric is None or  staging_metric > prod_metric:
        logging.info(f"Promoting staging model v{latest_staging.version} to Production.")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging.version,
            stage="Production",
            archive_existing_versions=True  # Archive old production model
        )
    else:
        logging.info("Staging model did not outperform production. No promotion done.")

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
        staging_version = register_model(model_name, model_info)
        promote_model_to_production(model_name, staging_version, metric_name="accuracy")
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()