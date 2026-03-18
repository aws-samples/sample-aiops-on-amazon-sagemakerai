"""MLflow utility functions for experiment tracking and model management."""

import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow_tracking(tracking_uri: str) -> None:
    """
    Set up MLflow tracking with the provided URI.

    Args:
        tracking_uri: MLflow tracking URI (can be ARN for SageMaker MLflow)
    """
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an MLflow experiment by name.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Experiment ID
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    return experiment_id


def register_model(model_name: str, run_id: str) -> None:
    """
    Register a model in MLflow Model Registry.

    Args:
        model_name: Name for the registered model
        run_id: MLflow run ID containing the model
    """
    model_uri = f"runs:/{run_id}/model"

    try:
        result = mlflow.register_model(model_uri, model_name)
        print(f"Model registered: {model_name} (version {result.version})")
    except Exception as e:
        print(f"Warning: Model registration failed: {e}")
        print(f"Model artifacts are still available at: {model_uri}")


def get_monitoring_experiment() -> str:
    """
    Get or create monitoring experiment.

    Returns:
        Monitoring experiment ID
    """
    from src.config.config import MLFLOW_MONITORING_EXPERIMENT_NAME
    return get_or_create_experiment(MLFLOW_MONITORING_EXPERIMENT_NAME)
