"""
SageMaker deployment with enhanced Athena logging.

Deploys models with inference_handler.py for comprehensive
inference tracking and monitoring.
"""

import json
import os
import pickle
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = _project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed, will use system environment variables only")

import mlflow

# Import SageMaker components
try:
    from sagemaker.serverless import ServerlessInferenceConfig
except ImportError:
    try:
        from sagemaker.core.inference_config import ServerlessInferenceConfig
    except ImportError:
        raise ImportError(
            "Could not import ServerlessInferenceConfig. "
            "Please ensure sagemaker SDK version 2.x is installed: "
            "pip install 'sagemaker>=2.210.0,<3.0.0'"
        )

try:
    from sagemaker.xgboost import XGBoostModel
    from sagemaker import image_uris
    from sagemaker.predictor import Predictor
except ImportError as e:
    raise ImportError(
        f"Could not import SageMaker v2.x modules: {e}\n"
        "Please install the correct SageMaker SDK version: "
        "pip install 'sagemaker>=2.210.0,<3.0.0'"
    )

from src.config.config import (
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    SAGEMAKER_EXEC_ROLE,
    SERVERLESS_MEMORY_SIZE,
    SERVERLESS_MAX_CONCURRENCY,
    ENABLE_ATHENA_LOGGING,
    INFERENCE_LOG_BATCH_SIZE,
    INFERENCE_LOG_FLUSH_INTERVAL,
    ATHENA_DATABASE,
    ATHENA_OUTPUT_S3,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_LOWER,
    LOW_CONFIDENCE_UPPER,
)
from src.utils.mlflow_utils import setup_mlflow_tracking
from src.utils.aws_utils import get_sagemaker_session, get_execution_role


def deploy(
    run_id: str,
    endpoint_name: str,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    memory_size_mb: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    enable_athena_logging: Optional[bool] = None,
) -> Predictor:
    """
    Deploy MLflow model to SageMaker with enhanced Athena logging.

    Args:
        run_id: MLflow run ID containing the model
        endpoint_name: Name for the SageMaker endpoint
        model_name: Optional model name (defaults to config)
        model_version: Optional model version tag
        memory_size_mb: Memory allocation (defaults to config)
        max_concurrency: Max concurrent invocations (defaults to config)
        enable_athena_logging: Enable Athena logging (defaults to config)

    Returns:
        SageMaker Predictor for the deployed endpoint
    """
    # Use defaults from config if not provided
    model_name = model_name or MLFLOW_MODEL_NAME
    model_version = model_version or run_id[:8]
    memory_size_mb = memory_size_mb or SERVERLESS_MEMORY_SIZE
    max_concurrency = max_concurrency or SERVERLESS_MAX_CONCURRENCY
    enable_athena_logging = enable_athena_logging if enable_athena_logging is not None else ENABLE_ATHENA_LOGGING

    print("=" * 80)
    print("SAGEMAKER DEPLOYMENT WITH ATHENA LOGGING")
    print("=" * 80)
    print(f"Endpoint: {endpoint_name}")
    print(f"Model: {model_name}")
    print(f"Run ID: {run_id}")
    print(f"Athena Logging: {'Enabled' if enable_athena_logging else 'Disabled'}")
    print("=" * 80)

    # Setup MLflow
    print("\nSetting up MLflow...")
    setup_mlflow_tracking(MLFLOW_TRACKING_URI)

    # Get SageMaker session and role
    print("Getting SageMaker session and execution role...")
    sagemaker_session = get_sagemaker_session()

    # Use config role if available, otherwise fallback to session detection
    if SAGEMAKER_EXEC_ROLE:
        role = SAGEMAKER_EXEC_ROLE
        print(f"Using execution role from config: {role}")
    else:
        role = get_execution_role(sagemaker_session=sagemaker_session)
        print(f"Detected execution role: {role}")

    # Create temporary directory for model packaging
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        mlflow_download_dir = temp_path / "mlflow_artifacts"
        mlflow_download_dir.mkdir()
        model_dir = temp_path / "model_package"
        model_dir.mkdir()

        print(f"\nDownloading model artifacts from MLflow run: {run_id}")

        # Download model artifacts from MLflow
        try:
            # Download the model
            model_uri = f"runs:/{run_id}/model"
            mlflow_model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, dst_path=str(mlflow_download_dir)
            )
            print(f"Downloaded model to: {mlflow_model_path}")

            # List all files in the downloaded directory
            mlflow_path = Path(mlflow_model_path)
            print(f"Contents of {mlflow_path}:")
            for item in mlflow_path.rglob("*"):
                if item.is_file():
                    print(f"  {item.relative_to(mlflow_path)}")

            # Find the model file
            possible_model_paths = [
                mlflow_path / "model.pkl",
                mlflow_path / "model.xgb",
                mlflow_path / "model.ubj",
                mlflow_path / "data" / "model.pkl",
                mlflow_path / "data" / "model.xgb",
                mlflow_path / "data" / "model.ubj",
            ]

            model_data_path = None
            for path in possible_model_paths:
                if path.exists():
                    model_data_path = path
                    print(f"Found model file: {path.relative_to(mlflow_path)}")
                    break

            if model_data_path is None:
                # Search recursively for any model file
                pkl_files = list(mlflow_path.rglob("*.pkl"))
                xgb_files = list(mlflow_path.rglob("*.xgb"))
                ubj_files = list(mlflow_path.rglob("*.ubj"))

                if pkl_files:
                    model_data_path = pkl_files[0]
                    print(f"Found model file (recursive search): {model_data_path.relative_to(mlflow_path)}")
                elif xgb_files:
                    model_data_path = xgb_files[0]
                    print(f"Found model file (recursive search): {model_data_path.relative_to(mlflow_path)}")
                elif ubj_files:
                    model_data_path = ubj_files[0]
                    print(f"Found model file (recursive search): {model_data_path.relative_to(mlflow_path)}")
                else:
                    raise FileNotFoundError(
                        f"Model file not found in {mlflow_model_path}\n"
                        f"Looked for: model.pkl, model.xgb, model.ubj in various locations"
                    )

            # Convert .ubj to JSON format for SageMaker compatibility
            if model_data_path.suffix == ".ubj":
                print("Converting .ubj model to JSON format for SageMaker compatibility...")
                import xgboost as xgb_lib

                bst = xgb_lib.Booster()
                bst.load_model(str(model_data_path))

                target_path = model_dir / "xgboost-model.json"
                bst.save_model(str(target_path))
                print(f"Converted and saved model as: xgboost-model.json")
            else:
                target_filename = "model" + model_data_path.suffix
                shutil.copy(model_data_path, model_dir / target_filename)
                print(f"Copied model file: {target_filename}")

        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

        # Download feature metadata
        try:
            feature_metadata_uri = f"runs:/{run_id}/feature_names.json"
            feature_metadata_path = mlflow.artifacts.download_artifacts(
                artifact_uri=feature_metadata_uri, dst_path=str(mlflow_download_dir)
            )
            shutil.copy(feature_metadata_path, model_dir / "feature_names.json")
            print("Downloaded feature_names.json")
        except Exception as e:
            print(f"Warning: Could not download feature_names.json: {e}")
            print("Creating default feature metadata...")
            feature_metadata = {
                "feature_names": [],
                "num_features": 0,
            }
            with open(model_dir / "feature_names.json", "w") as f:
                json.dump(feature_metadata, f)

        # Create model.tar.gz
        model_archive_path = temp_path / "model.tar.gz"
        print(f"\nCreating model archive: {model_archive_path}")

        with tarfile.open(model_archive_path, "w:gz") as tar:
            for file_path in model_dir.iterdir():
                tar.add(file_path, arcname=file_path.name)
                print(f"  Added: {file_path.name}")

        # Upload to S3
        print("\nUploading model archive to S3...")
        s3_bucket = sagemaker_session.default_bucket()
        s3_key_prefix = f"sg-xgboost-fraud-detection/{run_id}"
        model_data_s3 = sagemaker_session.upload_data(
            path=str(model_archive_path),
            bucket=s3_bucket,
            key_prefix=s3_key_prefix,
        )
        print(f"Model uploaded to: {model_data_s3}")

    # Create XGBoost Model with enhanced inference handler
    print("\nCreating SageMaker XGBoost model with enhanced inference handler...")
    inference_script_path = Path(__file__).parent.parent / "deployment" / "inference_handler.py"

    # Environment variables for inference logging
    environment = {
        'ENABLE_ATHENA_LOGGING': str(enable_athena_logging).lower(),
        'INFERENCE_LOG_BATCH_SIZE': str(INFERENCE_LOG_BATCH_SIZE),
        'INFERENCE_LOG_FLUSH_INTERVAL': str(INFERENCE_LOG_FLUSH_INTERVAL),
        'ENDPOINT_NAME': endpoint_name,
        'MODEL_VERSION': model_version,
        'MLFLOW_RUN_ID': run_id,
        'ATHENA_DATABASE': ATHENA_DATABASE,
        'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
        'HIGH_CONFIDENCE_THRESHOLD': str(HIGH_CONFIDENCE_THRESHOLD),
        'LOW_CONFIDENCE_LOWER': str(LOW_CONFIDENCE_LOWER),
        'LOW_CONFIDENCE_UPPER': str(LOW_CONFIDENCE_UPPER),
    }

    print("\nInference handler configuration:")
    for key, value in environment.items():
        print(f"  {key}: {value}")

    xgb_model = XGBoostModel(
        model_data=model_data_s3,
        role=role,
        entry_point=str(inference_script_path),
        framework_version="1.7-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
        env=environment,
    )

    # Configure serverless inference
    print(f"\nConfiguring serverless inference:")
    print(f"  Memory: {memory_size_mb} MB")
    print(f"  Max concurrency: {max_concurrency}")

    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=memory_size_mb,
        max_concurrency=max_concurrency,
    )

    # Check if endpoint already exists and delete if in failed state
    try:
        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = endpoint_desc['EndpointStatus']
        print(f"\nEndpoint '{endpoint_name}' already exists with status: {endpoint_status}")

        if endpoint_status == 'Failed':
            print("Deleting failed endpoint...")
            sagemaker_session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print("Waiting for endpoint deletion...")
            import time
            time.sleep(10)
    except sagemaker_session.sagemaker_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"Endpoint '{endpoint_name}' does not exist yet, will create new one")
        else:
            raise

    # Check if endpoint config exists and delete it
    try:
        sagemaker_session.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Endpoint configuration '{endpoint_name}' already exists, deleting...")
        sagemaker_session.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print("Endpoint configuration deleted")
    except sagemaker_session.sagemaker_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            pass
        else:
            raise

    # Deploy to endpoint
    print(f"\nDeploying to endpoint: {endpoint_name}")
    print("This may take several minutes...")

    predictor = xgb_model.deploy(
        endpoint_name=endpoint_name,
        serverless_inference_config=serverless_config,
    )

    print("\n" + "=" * 80)
    print("DEPLOYMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nEndpoint Name: {endpoint_name}")
    print(f"Endpoint Status: InService")
    print(f"Model Data: {model_data_s3}")
    print(f"Athena Logging: {'Enabled' if enable_athena_logging else 'Disabled'}")
    print(f"\nTo test the endpoint:")
    print(f"  python -m src.pipeline.test_endpoint --endpoint-name {endpoint_name}")

    return predictor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy fraud detection model to SageMaker")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    parser.add_argument("--model-version", help="Model version tag")
    parser.add_argument("--memory-size", type=int, help="Memory size in MB")
    parser.add_argument("--max-concurrency", type=int, help="Max concurrent invocations")
    parser.add_argument("--disable-athena-logging", action="store_true",
                       help="Disable Athena logging")

    args = parser.parse_args()

    # Deploy
    predictor = deploy(
        run_id=args.run_id,
        endpoint_name=args.endpoint_name,
        model_version=args.model_version,
        memory_size_mb=args.memory_size,
        max_concurrency=args.max_concurrency,
        enable_athena_logging=not args.disable_athena_logging,
    )

    print(f"\n✓ Deployment completed successfully!")
    print(f"  Endpoint: {args.endpoint_name}")
