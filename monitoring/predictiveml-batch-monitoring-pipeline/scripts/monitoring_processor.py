"""
SageMaker Processing script for automated data drift and data quality monitoring.

This script intentionally does NOT calculate model quality metrics.

Why:
    Data drift and data quality can be checked as soon as a new input file
    arrives. Model quality needs ground truth labels, and those labels often
    arrive later. Keeping those two concerns separate makes the automated
    drift pipeline easier for customers to plug into their own S3 data layout.

What this script does:
    1. Receives explicit S3 locations for baseline data and current data.
    2. Optionally resolves the latest CSV under a prefix when requested.
    3. Downloads both CSV files from S3.
    4. Runs Evidently DataDriftPreset and DataSummaryPreset.
    5. Logs metrics and report artifacts to SageMaker managed MLflow.
    6. Sends an SNS email notification when drift is detected.
    7. Writes a small summary JSON file for auditing or downstream steps.

Expected CSV shape:
    - Baseline CSV: feature columns with headers.
    - Current CSV: same feature columns with headers.

The explicit S3 file path is the default because it is the most predictable
integration pattern for customers. A scheduler, upstream data pipeline, or
orchestration tool can pass the exact file that should be monitored. Latest
file pickup remains available as an optional convenience for simple demos.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

# Install runtime dependencies inside the processing container. This keeps the
# notebook simple because the customer does not need to build a custom Docker
# image before trying the example. For production, building a pinned image is
# usually faster and more repeatable.
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
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_s3_uri(s3_uri):
    """
    Split an S3 URI into bucket and key parts.

    Example:
        s3://my-bucket/data/current/file.csv

    Returns:
        ("my-bucket", "data/current/file.csv")
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected an S3 URI starting with s3://, got: {s3_uri}")

    path_without_scheme = s3_uri.replace("s3://", "", 1)
    bucket, separator, key = path_without_scheme.partition("/")

    if not bucket or not separator:
        raise ValueError(f"Expected S3 URI with bucket and key, got: {s3_uri}")

    return bucket, key


def build_s3_uri(bucket, key):
    """Build a normal s3://bucket/key URI from separate bucket and key values."""
    return f"s3://{bucket}/{key}"


def find_latest_csv_s3_uri(s3_client, s3_prefix_uri):
    """
    Find the newest CSV file under an S3 prefix.

    This is useful for scheduled demo pipelines where each run should monitor
    the newest file in a landing folder. It is optional because in production
    many customers prefer the upstream job to pass the exact file path. Passing
    the exact file path avoids ambiguity when more than one file lands in the
    same folder at nearly the same time.
    """
    bucket, prefix = parse_s3_uri(s3_prefix_uri)

    # If the user passed s3://bucket/folder, treat it like a prefix. Adding a
    # trailing slash is not required by S3, but it helps avoid accidental
    # matches such as "folder-old/file.csv".
    if prefix and not prefix.endswith("/") and not prefix.lower().endswith(".csv"):
        prefix = prefix + "/"

    latest_key = None
    latest_modified = None
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.lower().endswith(".csv"):
                continue

            if latest_modified is None or item["LastModified"] > latest_modified:
                latest_key = key
                latest_modified = item["LastModified"]

    if latest_key is None:
        raise FileNotFoundError(f"No CSV files found under {s3_prefix_uri}")

    latest_uri = build_s3_uri(bucket, latest_key)
    logger.info("Latest CSV selected: %s modified at %s", latest_uri, latest_modified)
    return latest_uri


def resolve_input_csv_uri(s3_client, s3_uri, use_latest_file_pickup):
    """
    Return the exact CSV file URI that should be monitored.

    Default behavior:
        The URI must point directly to a CSV object. This is the most
        plug-and-play option for customers because the caller controls exactly
        which file is monitored.

    Optional behavior:
        If use_latest_file_pickup is true and the URI is a prefix, the script
        searches that prefix and picks the newest CSV by S3 LastModified time.
    """
    bucket, key = parse_s3_uri(s3_uri)

    if key.lower().endswith(".csv"):
        return build_s3_uri(bucket, key)

    if use_latest_file_pickup:
        return find_latest_csv_s3_uri(s3_client, s3_uri)

    raise ValueError(
        "The input S3 URI must point to a CSV file. "
        "Pass --use-latest-file-pickup true if you want to pass a prefix instead. "
        f"Received: {s3_uri}"
    )


def download_csv_from_s3(s3_client, s3_uri):
    """
    Download a CSV file from S3 and load it as a pandas DataFrame.

    A temporary local file keeps the code simple and avoids relying on pandas
    S3 plugins inside the processing container.
    """
    bucket, key = parse_s3_uri(s3_uri)

    with tempfile.NamedTemporaryFile(suffix=".csv") as local_file:
        logger.info("Downloading %s", s3_uri)
        s3_client.download_file(bucket, key, local_file.name)
        data = pd.read_csv(local_file.name)

    logger.info("Loaded %s with %d rows and %d columns", s3_uri, len(data), len(data.columns))
    return data


def safe_metric_name(text):
    """
    Convert a column name into an MLflow-safe metric name piece.

    MLflow metric names should be simple. This keeps feature-level metrics
    readable while avoiding characters that can cause tracking UI issues.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.:/ -]+", "_", str(text))
    return cleaned.strip() or "unknown"


def log_basic_quality_metrics(data, prefix):
    """
    Log simple data quality metrics that are easy to interpret in MLflow.

    Evidently produces the rich HTML report. These MLflow metrics provide a
    quick trend view without opening the report artifact.
    """
    row_count = len(data)
    column_count = len(data.columns)
    total_cells = row_count * column_count
    missing_cell_count = int(data.isna().sum().sum())
    duplicate_row_count = int(data.duplicated().sum())

    mlflow.log_metric(f"{prefix}.row_count", row_count)
    mlflow.log_metric(f"{prefix}.column_count", column_count)
    mlflow.log_metric(f"{prefix}.missing_cell_count", missing_cell_count)
    mlflow.log_metric(f"{prefix}.duplicate_row_count", duplicate_row_count)

    if total_cells > 0:
        mlflow.log_metric(f"{prefix}.missing_cell_share", missing_cell_count / total_cells)
    else:
        mlflow.log_metric(f"{prefix}.missing_cell_share", 0.0)

    # Log only columns that actually have missing values. This keeps MLflow from
    # being filled with hundreds of zero-valued metrics for wide datasets.
    missing_by_column = data.isna().sum()
    for column_name, missing_count in missing_by_column.items():
        if int(missing_count) > 0:
            mlflow.log_metric(
                f"{prefix}.missing:{safe_metric_name(column_name)}",
                int(missing_count),
            )


def extract_drift_summary(drift_results):
    """
    Pull a small alert-friendly drift summary out of Evidently's result dict.

    Evidently's full JSON is useful for analysis, but SNS alerts should be short
    and readable. This function extracts the count, share, and feature names
    that matter most during first triage.
    """
    metrics = drift_results.get("metrics", []) or []
    drifted_count = 0
    drifted_share = 0.0
    drifted_features = []

    for metric in metrics:
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {}) or {}
        value = metric.get("value")

        if metric_name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drifted_count = int(value.get("count", 0))
                drifted_share = float(value.get("share", 0.0))
            elif value is not None:
                drifted_count = int(value)

        if metric_name.startswith("ValueDrift"):
            column = config.get("column")
            threshold = config.get("threshold")

            if column is None or threshold is None:
                continue

            try:
                drift_score = float(value)
                drift_threshold = float(threshold)
            except (TypeError, ValueError):
                continue

            if drift_score > drift_threshold:
                drifted_features.append(
                    {
                        "column": column,
                        "drift_score": drift_score,
                        "threshold": drift_threshold,
                    }
                )

    return {
        "drift_detected": drifted_count > 0,
        "drifted_columns_count": drifted_count,
        "drifted_columns_share": drifted_share,
        "drifted_features": drifted_features,
    }


def log_drift_metrics(drift_results):
    """
    Log drift metrics from Evidently to MLflow.

    We log the overall drift count/share every run so trends are visible. For
    feature-level drift, we only log features that crossed a threshold to keep
    the experiment readable.
    """
    drift_summary = extract_drift_summary(drift_results)

    mlflow.log_metric("DriftedColumnsCount.count", drift_summary["drifted_columns_count"])
    mlflow.log_metric("DriftedColumnsCount.share", drift_summary["drifted_columns_share"])

    for feature in drift_summary["drifted_features"]:
        metric_name = f"ValueDrift:{safe_metric_name(feature['column'])}"
        mlflow.log_metric(metric_name, feature["drift_score"])


def send_drift_notification(
    sns_client,
    sns_topic_arn,
    drift_summary,
    mlflow_run_name,
    mlflow_experiment_name,
    baseline_source,
    current_source,
):
    """Send a readable data drift alert through Amazon SNS."""
    if drift_summary["drifted_features"]:
        feature_lines = "\n".join(
            (
                f"  - {item['column']}: "
                f"drift_score={item['drift_score']:.4f}, "
                f"threshold={item['threshold']:.4f}"
            )
            for item in drift_summary["drifted_features"]
        )
    else:
        feature_lines = "  - Evidently reported drift, but no feature-level scores were extracted."

    message = (
        "DATA DRIFT DETECTED\n"
        "==================================================\n\n"
        f"MLflow Experiment : {mlflow_experiment_name}\n"
        f"MLflow Run        : {mlflow_run_name}\n"
        f"Baseline Data     : {baseline_source}\n"
        f"Current Data      : {current_source}\n\n"
        f"Drifted Columns   : {drift_summary['drifted_columns_count']} "
        f"({drift_summary['drifted_columns_share']:.1%} of columns)\n\n"
        f"Drifted Features:\n{feature_lines}\n\n"
        "Open the MLflow run artifacts to review the full Evidently reports.\n"
    )

    sns_client.publish(
        TopicArn=sns_topic_arn,
        Subject=f"Data Drift Alert - {mlflow_experiment_name}",
        Message=message,
    )
    logger.info("SNS drift notification sent to %s", sns_topic_arn)


def parse_bool(text):
    """Parse simple true/false command-line values."""
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser(description="Run data drift and data quality monitoring.")
    parser.add_argument(
        "--baseline-data-s3-uri",
        required=True,
        help="S3 URI for baseline CSV, or a prefix when latest-file pickup is enabled.",
    )
    parser.add_argument(
        "--current-data-s3-uri",
        required=True,
        help="S3 URI for current CSV, or a prefix when latest-file pickup is enabled.",
    )
    parser.add_argument(
        "--use-latest-file-pickup",
        default="false",
        help="Set to true to pick the newest CSV under a provided S3 prefix.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        required=True,
        help="SageMaker managed MLflow tracking URI or app ARN.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        required=True,
        help="MLflow experiment that will receive monitoring runs.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default="",
        help="Optional run name. If omitted, a timestamped name is used.",
    )
    parser.add_argument(
        "--sns-topic-arn",
        default="",
        help="Optional SNS topic ARN for drift alert emails.",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", ""),
        help="AWS region for boto3 clients.",
    )
    args = parser.parse_args()

    boto_kwargs = {"region_name": args.region} if args.region else {}
    s3_client = boto3.client("s3", **boto_kwargs)
    sns_client = boto3.client("sns", **boto_kwargs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow_run_name = args.mlflow_run_name or f"data_drift_quality_monitoring_{timestamp}"
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)

    use_latest_file_pickup = parse_bool(args.use_latest_file_pickup)

    # Resolve exact file paths before downloading. This makes the summary and
    # MLflow parameters show the real files that were used, even when latest
    # pickup was enabled.
    baseline_csv_uri = resolve_input_csv_uri(
        s3_client,
        args.baseline_data_s3_uri,
        use_latest_file_pickup,
    )
    current_csv_uri = resolve_input_csv_uri(
        s3_client,
        args.current_data_s3_uri,
        use_latest_file_pickup,
    )

    reference_df = download_csv_from_s3(s3_client, baseline_csv_uri)
    current_df = download_csv_from_s3(s3_client, current_csv_uri)

    # Evidently expects the same feature columns in reference and current data.
    # Failing early with a clear message is better than producing a confusing
    # report when a current file has a wrong schema.
    reference_columns = list(reference_df.columns)
    current_columns = list(current_df.columns)
    if reference_columns != current_columns:
        raise ValueError(
            "Baseline and current CSV files must have the same columns in the same order. "
            f"Baseline columns: {reference_columns}. Current columns: {current_columns}."
        )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        logger.info("Started MLflow run %s with id %s", mlflow_run_name, run.info.run_id)

        mlflow.log_params(
            {
                "monitoring_type": "data_drift_and_quality",
                "baseline_data_s3_uri": baseline_csv_uri,
                "current_data_s3_uri": current_csv_uri,
                "use_latest_file_pickup": str(use_latest_file_pickup),
                "reference_data_rows": len(reference_df),
                "current_data_rows": len(current_df),
                "column_count": len(reference_df.columns),
                "monitoring_timestamp": timestamp,
            }
        )

        log_basic_quality_metrics(reference_df, "baseline_data_quality")
        log_basic_quality_metrics(current_df, "current_data_quality")

        data_definition = DataDefinition()
        reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
        current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

        logger.info("Running Evidently DataDriftPreset.")
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_snapshot = drift_report.run(
            reference_data=reference_dataset,
            current_data=current_dataset,
        )

        logger.info("Running Evidently DataSummaryPreset.")
        quality_report = Report(metrics=[DataSummaryPreset()])
        quality_snapshot = quality_report.run(
            reference_data=reference_dataset,
            current_data=current_dataset,
        )

        drift_html_path = os.path.join(output_dir, f"data_drift_report_{timestamp}.html")
        drift_json_path = os.path.join(output_dir, f"data_drift_report_{timestamp}.json")
        quality_html_path = os.path.join(output_dir, f"data_quality_report_{timestamp}.html")
        quality_json_path = os.path.join(output_dir, f"data_quality_report_{timestamp}.json")

        drift_snapshot.save_html(drift_html_path)
        drift_snapshot.save_json(drift_json_path)
        quality_snapshot.save_html(quality_html_path)
        quality_snapshot.save_json(quality_json_path)

        drift_results = drift_snapshot.dict()
        drift_summary = extract_drift_summary(drift_results)
        log_drift_metrics(drift_results)

        mlflow.log_artifact(drift_html_path, "evidently_data_drift")
        mlflow.log_artifact(drift_json_path, "evidently_data_drift")
        mlflow.log_artifact(quality_html_path, "evidently_data_quality")
        mlflow.log_artifact(quality_json_path, "evidently_data_quality")

        summary = {
            "mlflow_run_id": run.info.run_id,
            "mlflow_run_name": mlflow_run_name,
            "monitoring_type": "data_drift_and_quality",
            "baseline_data_s3_uri": baseline_csv_uri,
            "current_data_s3_uri": current_csv_uri,
            "use_latest_file_pickup": use_latest_file_pickup,
            "drift_detected": drift_summary["drift_detected"],
            "drifted_columns_count": drift_summary["drifted_columns_count"],
            "drifted_columns_share": drift_summary["drifted_columns_share"],
            "monitoring_timestamp": timestamp,
        }

        summary_path = os.path.join(output_dir, "monitoring_summary.json")
        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

        mlflow.log_artifact(summary_path, "monitoring_summary")

    if drift_summary["drift_detected"] and args.sns_topic_arn:
        send_drift_notification(
            sns_client=sns_client,
            sns_topic_arn=args.sns_topic_arn,
            drift_summary=drift_summary,
            mlflow_run_name=mlflow_run_name,
            mlflow_experiment_name=args.mlflow_experiment_name,
            baseline_source=baseline_csv_uri,
            current_source=current_csv_uri,
        )
    elif drift_summary["drift_detected"]:
        logger.warning("Drift detected, but no SNS topic ARN was provided.")
    else:
        logger.info("No data drift detected.")

    logger.info("Monitoring summary: %s", json.dumps(summary, indent=2))
    logger.info("Data drift and quality monitoring completed successfully.")


if __name__ == "__main__":
    main()
