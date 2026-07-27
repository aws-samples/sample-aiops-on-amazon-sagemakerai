"""
SageMaker Processing script for model quality monitoring.

This script is intentionally separate from monitoring_processor.py.

Why:
    Model quality needs two things that data drift does not need:
      1. Model predictions.
      2. Ground truth labels.

    In real production systems, labels can arrive hours, days, or weeks after
    the prediction was made. Keeping model quality in its own notebook and
    processing script lets teams run it when labels are available, while the
    data drift and quality pipeline can run immediately on new input files.

Expected CSV files:
    Predictions CSV:
        A CSV with a predicted label column, for example:
            prediction
            0
            1
            0

        It can also include a probability column, for example:
            prediction,prediction_proba
            0,0.13
            1,0.89
            0,0.22

    Ground truth CSV:
        A CSV with the true target label column, for example:
            target
            0
            1
            1

The rows must line up: row 1 in predictions must describe the same business
record as row 1 in ground truth. In production, teams usually keep an ID column
and join on that ID before running this processor.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime

# Install dependencies at runtime so the example works with the standard
# SageMaker SKLearn processing image. For production, use a custom image with
# pinned dependencies to reduce startup time.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "evidently>=0.7.20",
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow==3.4.0",
        "sagemaker-mlflow==0.2.0",
        "protobuf==3.20.3",
        "--no-cache-dir",
        "--ignore-installed",
    ]
)

import boto3
import mlflow
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_s3_uri(s3_uri):
    """Split s3://bucket/key into bucket and key."""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected an S3 URI starting with s3://, got: {s3_uri}")

    path_without_scheme = s3_uri.replace("s3://", "", 1)
    bucket, separator, key = path_without_scheme.partition("/")

    if not bucket or not separator:
        raise ValueError(f"Expected S3 URI with bucket and key, got: {s3_uri}")

    return bucket, key


def download_csv_from_s3(s3_client, s3_uri):
    """Download one CSV file from S3 and read it with pandas."""
    bucket, key = parse_s3_uri(s3_uri)

    with tempfile.NamedTemporaryFile(suffix=".csv") as local_file:
        logger.info("Downloading %s", s3_uri)
        s3_client.download_file(bucket, key, local_file.name)
        data = pd.read_csv(local_file.name)

    logger.info("Loaded %s with %d rows and %d columns", s3_uri, len(data), len(data.columns))
    return data


def require_column(data, column_name, file_description):
    """Fail early with a clear message if a required column is missing."""
    if column_name not in data.columns:
        raise ValueError(
            f"{file_description} is missing column '{column_name}'. "
            f"Available columns: {list(data.columns)}"
        )


def log_model_quality_metrics(eval_data, target_column, prediction_column, proba_column):
    """
    Calculate and log common binary classification metrics.

    These metrics are logged directly from scikit-learn because they are stable,
    familiar, and easy to compare across runs. Evidently still creates the rich
    HTML report artifact for deeper review.
    """
    target = eval_data[target_column]
    prediction = eval_data[prediction_column]

    mlflow.log_metric("Accuracy", accuracy_score(target, prediction))
    mlflow.log_metric("Precision", precision_score(target, prediction, zero_division=0))
    mlflow.log_metric("Recall", recall_score(target, prediction, zero_division=0))
    mlflow.log_metric("F1Score", f1_score(target, prediction, zero_division=0))

    if proba_column:
        try:
            mlflow.log_metric("ROC_AUC", roc_auc_score(target, eval_data[proba_column]))
        except ValueError as error:
            # ROC-AUC is undefined when the ground truth contains only one class.
            # The other metrics are still useful, so do not fail the whole run.
            logger.warning("ROC_AUC was not logged: %s", error)


def build_evaluation_dataframe(
    predictions_df,
    ground_truth_df,
    target_column,
    prediction_column,
    proba_column,
):
    """
    Build the small DataFrame that Evidently and scikit-learn will evaluate.

    This example aligns rows by position to keep the notebook straightforward.
    For customer production data, prefer joining predictions and labels by a
    stable record ID so late or missing labels do not shift rows accidentally.
    """
    require_column(predictions_df, prediction_column, "Predictions CSV")
    require_column(ground_truth_df, target_column, "Ground truth CSV")

    if proba_column:
        require_column(predictions_df, proba_column, "Predictions CSV")

    if len(predictions_df) != len(ground_truth_df):
        raise ValueError(
            "Predictions and ground truth must have the same number of rows for this example. "
            f"Predictions rows: {len(predictions_df)}. Ground truth rows: {len(ground_truth_df)}."
        )

    eval_data = pd.DataFrame()
    eval_data[target_column] = ground_truth_df[target_column].values
    eval_data[prediction_column] = predictions_df[prediction_column].values

    if proba_column:
        eval_data[proba_column] = predictions_df[proba_column].values

    return eval_data


def main():
    parser = argparse.ArgumentParser(description="Run binary classification model quality monitoring.")
    parser.add_argument(
        "--predictions-s3-uri",
        required=True,
        help="S3 URI for CSV containing model predictions.",
    )
    parser.add_argument(
        "--ground-truth-s3-uri",
        required=True,
        help="S3 URI for CSV containing true labels.",
    )
    parser.add_argument(
        "--target-column",
        default="target",
        help="Column in the ground truth CSV containing true labels.",
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Column in the predictions CSV containing predicted labels.",
    )
    parser.add_argument(
        "--prediction-proba-column",
        default="prediction_proba",
        help="Optional probability column in predictions CSV. Leave blank if unavailable.",
    )
    parser.add_argument(
        "--positive-label",
        default="1",
        help="Positive class label for binary classification reports.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        required=True,
        help="SageMaker managed MLflow tracking URI or app ARN.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        required=True,
        help="MLflow experiment that will receive model quality runs.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default="",
        help="Optional run name. If omitted, a timestamped name is used.",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", ""),
        help="AWS region for boto3 clients.",
    )
    args = parser.parse_args()

    boto_kwargs = {"region_name": args.region} if args.region else {}
    s3_client = boto3.client("s3", **boto_kwargs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow_run_name = args.mlflow_run_name or f"model_quality_monitoring_{timestamp}"
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)

    proba_column = args.prediction_proba_column.strip()
    if not proba_column:
        proba_column = None

    predictions_df = download_csv_from_s3(s3_client, args.predictions_s3_uri)
    ground_truth_df = download_csv_from_s3(s3_client, args.ground_truth_s3_uri)

    eval_data = build_evaluation_dataframe(
        predictions_df=predictions_df,
        ground_truth_df=ground_truth_df,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        proba_column=proba_column,
    )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        logger.info("Started MLflow run %s with id %s", mlflow_run_name, run.info.run_id)

        mlflow.log_params(
            {
                "monitoring_type": "model_quality",
                "predictions_s3_uri": args.predictions_s3_uri,
                "ground_truth_s3_uri": args.ground_truth_s3_uri,
                "target_column": args.target_column,
                "prediction_column": args.prediction_column,
                "prediction_proba_column": proba_column or "",
                "row_count": len(eval_data),
                "monitoring_timestamp": timestamp,
            }
        )

        log_model_quality_metrics(
            eval_data=eval_data,
            target_column=args.target_column,
            prediction_column=args.prediction_column,
            proba_column=proba_column,
        )

        # Tell Evidently which columns represent the true labels and predictions.
        # ClassificationPreset uses this schema to build the confusion matrix and
        # classification quality report.
        data_definition = DataDefinition(
            classification=[
                BinaryClassification(
                    target=args.target_column,
                    prediction_labels=args.prediction_column,
                    pos_label=int(args.positive_label),
                )
            ]
        )

        eval_dataset = Dataset.from_pandas(eval_data, data_definition=data_definition)
        classification_report = Report(metrics=[ClassificationPreset()])
        classification_snapshot = classification_report.run(
            reference_data=None,
            current_data=eval_dataset,
        )

        report_html_path = os.path.join(output_dir, f"model_quality_report_{timestamp}.html")
        report_json_path = os.path.join(output_dir, f"model_quality_report_{timestamp}.json")
        summary_path = os.path.join(output_dir, "model_quality_summary.json")

        classification_snapshot.save_html(report_html_path)
        classification_snapshot.save_json(report_json_path)

        summary = {
            "mlflow_run_id": run.info.run_id,
            "mlflow_run_name": mlflow_run_name,
            "monitoring_type": "model_quality",
            "predictions_s3_uri": args.predictions_s3_uri,
            "ground_truth_s3_uri": args.ground_truth_s3_uri,
            "row_count": len(eval_data),
            "monitoring_timestamp": timestamp,
        }

        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

        mlflow.log_artifact(report_html_path, "evidently_model_quality")
        mlflow.log_artifact(report_json_path, "evidently_model_quality")
        mlflow.log_artifact(summary_path, "model_quality_summary")

    logger.info("Model quality summary: %s", json.dumps(summary, indent=2))
    logger.info("Model quality monitoring completed successfully.")


if __name__ == "__main__":
    main()
