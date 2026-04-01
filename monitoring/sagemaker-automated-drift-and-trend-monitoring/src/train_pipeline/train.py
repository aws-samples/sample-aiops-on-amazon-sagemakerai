"""
SageMaker-focused training script with Athena integration.

Supports dual training modes:
- local: Train locally with data from Athena or CSV
- sagemaker: Submit SageMaker training job

Example usage:
    # Local training with Athena data
    python -m src.train_pipeline.train --training-mode local --data-source athena

    # SageMaker training job
    python -m src.train_pipeline.train --training-mode sagemaker --instance-type ml.m5.xlarge
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Any
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc as sklearn_auc,
)
from xgboost import XGBClassifier

from src.config.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    CSV_TRAINING_DATA,
    TEST_SIZE,
    RANDOM_STATE,
    XGBOOST_PARAMS,
    ATHENA_TRAINING_TABLE,
    S3_TRAINING_DATA_EXPORT,
    SAGEMAKER_EXEC_ROLE,
    SAGEMAKER_TRAINING_INSTANCE,
    SAGEMAKER_TRAINING_VOLUME_SIZE,
    TRAINING_FEATURES,
    TARGET_COLUMN,
    DATA_S3_BUCKET,
    EVIDENTLY_BASELINE_SAMPLES,
    EVIDENTLY_BASELINE_S3_PREFIX,
)
from src.train_pipeline.athena.athena_client import AthenaClient
from src.utils.mlflow_utils import setup_mlflow_tracking, get_or_create_experiment

logger = logging.getLogger(__name__)


def load_data_from_athena(
    table_name: str = ATHENA_TRAINING_TABLE,
    filters: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, list]:
    """
    Load and split training data from Athena.

    Args:
        table_name: Athena table name
        filters: Optional SQL WHERE clause
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scale_pos_weight, feature_names)
    """
    logger.info(f"Loading data from Athena table: {table_name}")

    # Initialize Athena client
    client = AthenaClient()

    # Read data
    df = client.read_table(table_name, filters=filters)

    logger.info(f"Loaded {len(df)} rows from Athena")

    # Prepare features and target
    feature_columns = [col for col in TRAINING_FEATURES if col in df.columns]
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    # One-hot encode customer_gender if present
    if 'customer_gender' in X.columns:
        X = pd.get_dummies(X, columns=['customer_gender'], drop_first=True)

    # Calculate scale_pos_weight for imbalanced data
    fraud_count = y.sum()
    non_fraud_count = len(y) - fraud_count
    scale_pos_weight = non_fraud_count / fraud_count if fraud_count > 0 else 1.0

    logger.info(f"Class distribution: {non_fraud_count} non-fraud, {fraud_count} fraud")
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scale_pos_weight, list(X.columns)


def load_data_from_csv(
    csv_path: Path = CSV_TRAINING_DATA,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, list]:
    """
    Load and split training data from CSV (fallback mode).

    Args:
        csv_path: Path to CSV file
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scale_pos_weight, feature_names)
    """
    logger.info(f"Loading data from CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Prepare features and target
    feature_columns = [col for col in TRAINING_FEATURES if col in df.columns]
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    # One-hot encode customer_gender if present
    if 'customer_gender' in X.columns:
        X = pd.get_dummies(X, columns=['customer_gender'], drop_first=True)

    # Calculate scale_pos_weight
    fraud_count = y.sum()
    non_fraud_count = len(y) - fraud_count
    scale_pos_weight = non_fraud_count / fraud_count if fraud_count > 0 else 1.0

    logger.info(f"Class distribution: {non_fraud_count} non-fraud, {fraud_count} fraud")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scale_pos_weight, list(X.columns)


def train_local(
    data_source: str = "athena",
    data_path: Optional[str] = None,
    athena_table: str = ATHENA_TRAINING_TABLE,
    athena_filter: Optional[str] = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    **xgboost_params,
) -> Tuple[str, XGBClassifier]:
    """
    Train model locally with Athena or CSV data.

    Args:
        data_source: 'athena' or 'csv'
        data_path: Path to CSV (if data_source='csv')
        athena_table: Athena table name (if data_source='athena')
        athena_filter: SQL WHERE clause for filtering
        experiment_name: MLflow experiment name
        **xgboost_params: Additional XGBoost parameters

    Returns:
        Tuple of (run_id, trained_model)
    """
    logger.info("Starting local training")

    # Setup MLflow
    setup_mlflow_tracking(MLFLOW_TRACKING_URI)
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Load data
    if data_source == "athena":
        X_train, X_test, y_train, y_test, scale_pos_weight, feature_names = load_data_from_athena(
            table_name=athena_table,
            filters=athena_filter,
        )
    else:
        csv_path = Path(data_path) if data_path else CSV_TRAINING_DATA
        X_train, X_test, y_train, y_test, scale_pos_weight, feature_names = load_data_from_csv(
            csv_path=csv_path,
        )

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_param("data_source", data_source)
        mlflow.log_param("training_mode", "local")
        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Enable autologging
        mlflow.xgboost.autolog(log_models=True, log_datasets=True)

        # Prepare parameters
        params = XGBOOST_PARAMS.copy()
        params["scale_pos_weight"] = scale_pos_weight
        params.update(xgboost_params)

        logger.info(f"Training with parameters: {params}")

        # Train model
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
        pr_auc = sklearn_auc(recall, precision)

        logger.info(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        # Log custom metrics
        mlflow.log_metric("roc_auc_test", roc_auc)
        mlflow.log_metric("pr_auc_test", pr_auc)

        # Log feature names
        mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

        # Register model
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)

        logger.info(f"Model registered: {MLFLOW_MODEL_NAME}")

        # Save baseline data for drift detection monitoring
        logger.info("Saving baseline data for monitoring...")
        try:
            # Sample training data as baseline reference
            baseline_sample_size = min(EVIDENTLY_BASELINE_SAMPLES, len(X_train))
            baseline_indices = np.random.choice(
                len(X_train), size=baseline_sample_size, replace=False
            )

            # Create baseline DataFrame with features and target
            baseline_df = X_train.iloc[baseline_indices].copy()
            baseline_df[TARGET_COLUMN] = y_train.iloc[baseline_indices].values

            # Save to S3 as baseline for drift detection
            import boto3 as _boto3
            _s3 = _boto3.client("s3")
            _baseline_key = f"{EVIDENTLY_BASELINE_S3_PREFIX}{run_id}/baseline.csv"
            _csv_buf = baseline_df.to_csv(index=False)
            _s3.put_object(Bucket=DATA_S3_BUCKET, Key=_baseline_key, Body=_csv_buf)
            baseline_s3_path = f"s3://{DATA_S3_BUCKET}/{_baseline_key}"

            # Log baseline metadata to MLflow
            mlflow.set_tag("baseline_data_path", baseline_s3_path)
            mlflow.log_param("baseline_sample_size", len(baseline_df))
            mlflow.log_param("baseline_fraud_rate", baseline_df[TARGET_COLUMN].mean())

            logger.info(f"✓ Baseline data saved: {baseline_s3_path}")

        except Exception as e:
            logger.warning(f"Failed to save baseline data (non-critical): {e}")
            logger.warning("  Monitoring with Evidently will require manual baseline setup")

        return run_id, model


def train_sagemaker(
    data_source: str = "athena",
    instance_type: str = SAGEMAKER_TRAINING_INSTANCE,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    **xgboost_params,
) -> Tuple[str, str]:
    """
    Submit SageMaker training job.

    Args:
        data_source: 'athena' or 'csv'
        instance_type: SageMaker instance type
        experiment_name: MLflow experiment name
        **xgboost_params: Additional XGBoost parameters

    Returns:
        Tuple of (run_id, s3_model_path)

    Note:
        This is a stub that would need full SageMaker integration.
        The complete implementation would:
        1. Export data from Athena to S3
        2. Create SageMaker estimator
        3. Submit training job
        4. Wait for completion
        5. Register model in MLflow
    """
    logger.warning("SageMaker training mode is not fully implemented yet")
    logger.info("To complete this implementation, you would need to:")
    logger.info("  1. Export training data from Athena to S3 as Parquet")
    logger.info("  2. Create SageMaker XGBoost estimator with custom entry point")
    logger.info("  3. Configure environment variables for MLflow tracking")
    logger.info("  4. Submit training job and wait for completion")
    logger.info("  5. Download model artifacts and register in MLflow")

    raise NotImplementedError("SageMaker training mode not yet implemented")


def train(
    training_mode: str = "local",
    data_source: str = "athena",
    data_path: Optional[str] = None,
    athena_table: str = ATHENA_TRAINING_TABLE,
    athena_filter: Optional[str] = None,
    instance_type: str = SAGEMAKER_TRAINING_INSTANCE,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    **xgboost_params,
) -> Tuple[str, Any]:
    """
    Train XGBoost model with Athena data and optional SageMaker job.

    Args:
        training_mode: 'local' or 'sagemaker'
        data_source: 'athena' or 'csv'
        data_path: Path to CSV (if using CSV)
        athena_table: Athena table name
        athena_filter: SQL WHERE clause
        instance_type: SageMaker instance type
        experiment_name: MLflow experiment name
        **xgboost_params: Additional XGBoost parameters

    Returns:
        Tuple of (run_id, model_or_path)
    """
    print("=" * 80)
    print("SAGEMAKER FRAUD DETECTION - TRAINING")
    print("=" * 80)
    print(f"Training Mode: {training_mode}")
    print(f"Data Source: {data_source}")
    print(f"Experiment: {experiment_name}")
    print("=" * 80 + "\n")

    if training_mode == "local":
        return train_local(
            data_source=data_source,
            data_path=data_path,
            athena_table=athena_table,
            athena_filter=athena_filter,
            experiment_name=experiment_name,
            **xgboost_params,
        )
    elif training_mode == "sagemaker":
        return train_sagemaker(
            data_source=data_source,
            instance_type=instance_type,
            experiment_name=experiment_name,
            **xgboost_params,
        )
    else:
        raise ValueError(f"Invalid training_mode: {training_mode}. Must be 'local' or 'sagemaker'")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--training-mode", choices=["local", "sagemaker"], default="local")
    parser.add_argument("--data-source", choices=["athena", "csv"], default="athena")
    parser.add_argument("--data-path", type=str, help="Path to CSV file")
    parser.add_argument("--athena-table", default=ATHENA_TRAINING_TABLE)
    parser.add_argument("--instance-type", default=SAGEMAKER_TRAINING_INSTANCE)

    args = parser.parse_args()

    # Train
    run_id, model = train(
        training_mode=args.training_mode,
        data_source=args.data_source,
        data_path=args.data_path,
        athena_table=args.athena_table,
        instance_type=args.instance_type,
    )

    print(f"\n✓ Training completed!")
    print(f"  Run ID: {run_id}")
    print(f"  Model: {MLFLOW_MODEL_NAME}")
