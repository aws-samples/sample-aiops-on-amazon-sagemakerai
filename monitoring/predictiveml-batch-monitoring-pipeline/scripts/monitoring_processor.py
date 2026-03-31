"""
SageMaker Processing Job script for automated data drift and data quality monitoring.

This script is executed inside a SageMaker Processing container as part of an automated
monitoring pipeline. It performs the following workflow:

1. Loads baseline (reference) data from training dataset
2. Loads current production data with feature headers for drift comparison
3. Loads prediction outputs from the batch transform job (optional for model quality)
4. Runs Evidently DataDriftPreset to detect feature distribution changes
5. Runs Evidently DataSummaryPreset to assess data quality
6. Logs all metrics and report artifacts to SageMaker AI MLflow for tracking
7. Sends SNS email notification if drift exceeds thresholds
8. Saves HTML/JSON reports to S3 output for auditing and review

Expected Input Channels:
- /opt/ml/processing/baseline: Training/reference data with headers (CSV)
- /opt/ml/processing/current: Inference input data WITHOUT headers (CSV)
- /opt/ml/processing/current_headers: Inference input data WITH headers (CSV)
- /opt/ml/processing/predictions: Batch transform prediction outputs (CSV)

Output Channel:
- /opt/ml/processing/output: Evidently reports and monitoring summary (HTML/JSON)
"""
import os
import sys

os.system(f'{sys.executable} -m pip install -U evidently>=0.7.20 pandas numpy scikit-learn mlflow==3.1.4 sagemaker-mlflow==0.2.0 s3fs protobuf==3.20.3 --no-cache-dir --ignore-installed')

import argparse
import json
import logging
import tempfile
from datetime import datetime

import boto3
import mlflow
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def parse_s3_uri(uri: str):
    """
    Parse an S3 URI into bucket and key components.

    Args:
        uri: S3 URI in format s3://bucket-name/path/to/object

    Returns:
        tuple: (bucket_name, key_path)

    Example:
        parse_s3_uri('s3://my-bucket/data/file.csv') -> ('my-bucket', 'data/file.csv')
    """
    path = uri.replace("s3://", "")
    bucket, _, key = path.partition("/")
    return bucket, key


def get_latest_csv_key(bucket: str, prefix: str) -> str:
    """
    Find the most recently modified CSV file in an S3 prefix.

    This enables the monitoring pipeline to automatically pick up the latest data
    without hardcoding filenames. Useful for scheduled monitoring runs.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix/folder to search

    Returns:
        str: S3 key of the most recent CSV file

    Raises:
        FileNotFoundError: If no CSV files are found in the specified location
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    latest_key = None
    latest_modified = None

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".csv") and not key.endswith("/"):
                if latest_modified is None or obj["LastModified"] > latest_modified:
                    latest_modified = obj["LastModified"]
                    latest_key = key

    if latest_key is None:
        raise FileNotFoundError(f"No CSV files found in s3://{bucket}/{prefix}")
    logger.info("Latest CSV: s3://%s/%s  (modified %s)", bucket, latest_key, latest_modified)
    return latest_key


def download_csv(bucket: str, key: str) -> pd.DataFrame:
    """
    Download a CSV file from S3 and load it into a pandas DataFrame.

    Args:
        bucket: S3 bucket name
        key: S3 object key (full path)

    Returns:
        pd.DataFrame: Loaded data with all columns and rows

    Note:
        Uses a temporary file to avoid memory issues with large datasets.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        s3_client.download_file(bucket, key, tmp.name)
        df = pd.read_csv(tmp.name)
    logger.info("Loaded %d rows x %d cols from s3://%s/%s", len(df), len(df.columns), bucket, key)
    return df


def extract_drift_summary(drift_results: dict) -> dict:
    """
    Extract human-readable drift summary from Evidently report results.

    Parses the Evidently report dictionary to extract:
    - Number and percentage of drifted columns
    - List of specific features that exceeded drift thresholds
    - Drift scores for each drifted feature

    Args:
        drift_results: Dictionary output from Evidently Report.dict()

    Returns:
        dict: Summary containing drift_detected (bool), drifted_columns_count (int),
              drifted_columns_share (float), and drifted_features (list of dicts)

    Note:
        This summary is used for SNS alerting and logging to downstream systems.
    """
    metrics = drift_results.get("metrics", []) or []
    drifted_count = 0
    drifted_share = 0.0
    drifted_features = []

    for m in metrics:
        metric_name = m.get("metric_name", "")
        cfg = m.get("config", {}) or {}
        val = m.get("value")

        if metric_name.startswith("DriftedColumnsCount"):
            if isinstance(val, dict):
                drifted_count = int(val.get("count", 0))
                drifted_share = float(val.get("share", 0.0))

        if metric_name.startswith("ValueDrift"):
            threshold = cfg.get("threshold")
            column = cfg.get("column")
            if threshold is None or column is None:
                continue
            try:
                numeric_val = float(val)
            except (TypeError, ValueError):
                continue
            if numeric_val > float(threshold):
                drifted_features.append({
                    "column": column,
                    "drift_score": numeric_val,
                    "threshold": float(threshold),
                })

    return {
        "drift_detected": drifted_count > 0,
        "drifted_columns_count": drifted_count,
        "drifted_columns_share": drifted_share,
        "drifted_features": drifted_features,
    }


def log_drift_metrics(drift_results: dict):
    """
    Extract and log drift metrics from Evidently report to MLflow.

    Logs two types of metrics:
    1. DriftedColumnsCount: Overall count and percentage of drifted features
    2. ValueDrift per feature: Individual drift scores for features exceeding threshold

    Args:
        drift_results: Dictionary output from Evidently Report.dict()

    MLflow Metrics Created:
        - DriftedColumnsCount.count: Number of features with detected drift
        - DriftedColumnsCount.share: Percentage of features with drift (0.0 to 1.0)
        - ValueDrift:<feature_name>: Drift score for each drifted feature

    Note:
        Only features exceeding their configured drift threshold are logged to
        reduce metric clutter. This allows tracking trends over time in MLflow.
    """
    metrics = drift_results.get("metrics", []) or []
    drifted_columns_logged = False

    for m in metrics:
        metric_name = m.get("metric_name", "")
        cfg = m.get("config", {}) or {}
        val = m.get("value")

        # Log overall drift count and percentage
        if metric_name.startswith("DriftedColumnsCount"):
            drifted_columns_logged = True
            if isinstance(val, dict):
                count = float(val.get("count", 0.0))
                share = float(val.get("share", 0.0))
            else:
                count = float(val) if val is not None else 0.0
                share = 0.0
            mlflow.log_metric("DriftedColumnsCount.count", count)
            mlflow.log_metric("DriftedColumnsCount.share", share)
            continue

        # Log per-feature drift scores (only if exceeds threshold)
        if metric_name.startswith("ValueDrift"):
            threshold = cfg.get("threshold")
            column = cfg.get("column")
            if threshold is None or column is None:
                continue
            try:
                numeric_val = float(val)
            except (TypeError, ValueError):
                continue
            # Only log if drift detected (exceeds threshold)
            if numeric_val > float(threshold):
                mlflow.log_metric(f"ValueDrift:{column}", numeric_val)

    # Ensure we always log drift count, even if zero
    if not drifted_columns_logged:
        mlflow.log_metric("DriftedColumnsCount.count", 0.0)


def send_drift_notification(sns_topic_arn, drift_summary, mlflow_run_name,
                            mlflow_experiment_name, baseline_source, production_source):
    """
    Send a formatted drift alert email via Amazon SNS.

    Constructs a detailed email message with:
    - MLflow run information for investigation
    - Data source locations
    - Number and percentage of drifted features
    - Specific features with drift scores

    Args:
        sns_topic_arn: ARN of the SNS topic to publish to
        drift_summary: Dictionary containing drift detection results
        mlflow_run_name: Name of the MLflow run for traceability
        mlflow_experiment_name: Name of the MLflow experiment
        baseline_source: S3 location or identifier of baseline data
        production_source: S3 location or identifier of production data

    Note:
        Email subscribers must confirm their subscription before receiving alerts.
        Check the SNS subscription status if alerts are not being delivered.
    """
    feature_lines = "\n".join(
        f"  - {f['column']}: drift_score={f['drift_score']:.4f} (threshold={f['threshold']:.4f})"
        for f in drift_summary["drifted_features"]
    )

    message = (
        f"DATA DRIFT DETECTED\n"
        f"{'=' * 50}\n\n"
        f"MLflow Experiment : {mlflow_experiment_name}\n"
        f"MLflow Run        : {mlflow_run_name}\n"
        f"Baseline Data     : {baseline_source}\n"
        f"Production Data   : {production_source}\n\n"
        f"Drifted Columns   : {drift_summary['drifted_columns_count']} "
        f"({drift_summary['drifted_columns_share']:.1%} of features)\n\n"
        f"Drifted Features:\n{feature_lines}\n\n"
        f"Review the full Evidently reports in the MLflow artifacts.\n"
    )

    sns_client.publish(
        TopicArn=sns_topic_arn,
        Subject=f"Data Drift Alert - {mlflow_experiment_name}",
        Message=message,
    )
    logger.info("Drift notification sent to SNS topic %s", sns_topic_arn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for the monitoring processing job.

    Workflow:
    1. Parse command-line arguments from SageMaker Pipeline
    2. Initialize AWS clients (S3, SNS)
    3. Load baseline, current, and prediction data from processing input channels
    4. Run Evidently drift and quality reports
    5. Log metrics and artifacts to MLflow
    6. Send SNS alert if drift detected
    7. Save summary JSON for downstream pipeline steps
    """
    # ------------------------------------------------------------------
    # Step 1: Parse Arguments and Initialize Clients
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Evidently-based data drift monitoring script")
    parser.add_argument("--baseline-filename", type=str, required=True,
                        help="Filename of baseline/reference data (with headers)")
    parser.add_argument("--input-filename", type=str, required=True,
                        help="Filename of current inference input data (without headers)")
    parser.add_argument("--input-headers-filename", type=str, required=True,
                        help="Filename of current inference input data (with headers for drift analysis)")
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True,
                        help="MLflow tracking server URI (SageMaker MLflow App ARN)")
    parser.add_argument("--mlflow-experiment-name", type=str, required=True,
                        help="MLflow experiment name for logging")
    parser.add_argument("--mlflow-run-name", type=str, default="",
                        help="Optional MLflow run name (auto-generated if not provided)")
    parser.add_argument("--sns-topic-arn", type=str, default="",
                        help="SNS topic ARN for drift alerts (optional)")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_DEFAULT_REGION", ""),
                        help="AWS region for boto3 clients")
    args = parser.parse_args()

    # Initialize AWS clients (S3 for data access, SNS for alerts)
    global s3_client, sns_client
    boto_kwargs = {"region_name": args.region} if args.region else {}
    s3_client = boto3.client("s3", **boto_kwargs)
    sns_client = boto3.client("sns", **boto_kwargs)

    # Generate timestamp for unique run identification
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mlflow_run_name = args.mlflow_run_name or f"data_drift_quality_monitoring_{timestamp}"

    # SageMaker Processing output directory (uploaded to S3 at job completion)
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2: Initialize MLflow Tracking
    # ------------------------------------------------------------------
    # Connect to the same MLflow tracking server used in the training notebook
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s (id=%s)", mlflow_run_name, run_id)

        # ------------------------------------------------------------------
        # Step 3: Load Data from Processing Input Channels
        # ------------------------------------------------------------------
        # SageMaker Processing mounts S3 data to local paths:
        # - /opt/ml/processing/baseline: Reference data from training (with headers)
        # - /opt/ml/processing/current_headers: Current data WITH headers (for drift analysis)
        # - /opt/ml/processing/current: Current data WITHOUT headers (used for inference)
        # - /opt/ml/processing/predictions: Batch transform outputs (predictions)

        baseline_filename = args.baseline_filename
        current_filename = args.input_filename  # Without headers (used by batch transform)
        current_headers_filename = args.input_headers_filename  # With headers (for Evidently)

        logger.info("Loading baseline data: %s", baseline_filename)
        reference_df = pd.read_csv(f'/opt/ml/processing/baseline/{baseline_filename}')

        logger.info("Loading current data with headers: %s", current_headers_filename)
        current_df = pd.read_csv(f'/opt/ml/processing/current_headers/{current_headers_filename}')

        logger.info("Loading predictions: %s.out", current_filename)
        predictions_df = pd.read_csv(f'/opt/ml/processing/predictions/{current_filename}.out', header=None)

        # NOTE: predictions_df can be used for model quality reports if ground truth labels are available
        # For this pipeline, we focus on data drift and data quality monitoring
        # To add model quality monitoring, merge predictions_df with actual labels and use
        # Evidently's ClassificationPreset or RegressionPreset

        # ------------------------------------------------------------------
        # Step 4: Run Evidently Monitoring Reports
        # ------------------------------------------------------------------
        # Wrap pandas DataFrames in Evidently Dataset objects
        # DataDefinition helps Evidently understand column types and structure
        data_definition = DataDefinition()  # Use default auto-detection since we have headers
        reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
        current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

        # Run DataDriftPreset: Detects distribution changes in features
        # Uses statistical tests (KS test, Chi-square) to compare reference vs current data
        logger.info("Running Evidently DataDriftPreset report...")
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_snapshot = drift_report.run(
            reference_data=reference_dataset,
            current_data=current_dataset,
        )

        # Run DataSummaryPreset: Assesses data quality issues
        # Identifies missing values, correlations, duplicates, and anomalies
        logger.info("Running Evidently DataSummaryPreset report...")
        quality_report = Report(metrics=[DataSummaryPreset()])
        quality_snapshot = quality_report.run(
            reference_data=reference_dataset,
            current_data=current_dataset,
        )

        # ------------------------------------------------------------------
        # Step 5: Save Reports to Local Output Directory
        # ------------------------------------------------------------------
        # Reports are saved locally first, then SageMaker automatically uploads
        # to S3 at the end of the processing job
        # Generate timestamped filenames for reports
        drift_html = os.path.join(output_dir, f"data_drift_report_{timestamp}.html")
        drift_json = os.path.join(output_dir, f"data_drift_report_{timestamp}.json")
        quality_html = os.path.join(output_dir, f"data_quality_report_{timestamp}.html")

        # Save Evidently reports in multiple formats
        drift_snapshot.save_html(drift_html)  # Interactive HTML for human review
        drift_snapshot.save_json(drift_json)  # Structured JSON for programmatic access
        quality_snapshot.save_html(quality_html)  # Data quality HTML report
        logger.info("Reports saved to %s", output_dir)

        # ------------------------------------------------------------------
        # Step 6: Log Metadata, Metrics, and Artifacts to MLflow
        # ------------------------------------------------------------------
        # Log configuration and context as parameters
        mlflow.log_params({
            "monitoring_type": "data_drift_and_quality",
            "baseline_data_source": f"{baseline_filename}",
            "current_data_source": f"{current_headers_filename}",
            "reference_data_rows": len(reference_df),
            "current_data_rows": len(current_df),
            "monitoring_timestamp": timestamp,
        })

        # Extract and log numeric drift metrics
        drift_results = drift_snapshot.dict()
        log_drift_metrics(drift_results)

        # Log HTML/JSON reports as MLflow artifacts
        # These will be accessible in the MLflow UI for review
        mlflow.log_artifact(drift_html, "evidently_report_data_drift_html")
        mlflow.log_artifact(drift_json, "evidently_report_data_drift_json")
        mlflow.log_artifact(quality_html, "evidently_report_data_quality")

        logger.info("All metrics and artifacts logged to MLflow run %s", run_id)

    # ------------------------------------------------------------------
    # Step 7: Send SNS Alert if Drift Detected
    # ------------------------------------------------------------------
    # This happens AFTER the MLflow run completes to ensure all data is logged
    # even if the SNS notification fails
    # Extract human-readable drift summary from Evidently results
    drift_summary = extract_drift_summary(drift_results)

    # Send email alert via SNS if drift was detected and SNS topic is configured
    if drift_summary["drift_detected"] and args.sns_topic_arn:
        logger.info("Drift detected! Sending SNS notification...")
        send_drift_notification(
            sns_topic_arn=args.sns_topic_arn,
            drift_summary=drift_summary,
            mlflow_run_name=mlflow_run_name,
            mlflow_experiment_name=args.mlflow_experiment_name,
            baseline_source=f"{baseline_filename}",
            production_source=f"{current_headers_filename}",
        )
    elif drift_summary["drift_detected"]:
        logger.warning("Drift detected but no SNS topic ARN provided; skipping notification.")
    else:
        logger.info("No data drift detected; no notification sent.")

    # ------------------------------------------------------------------
    # Step 8: Write Monitoring Summary for Downstream Pipeline Steps
    # ------------------------------------------------------------------
    # This JSON file can be consumed by downstream steps in the pipeline
    # (e.g., conditional model retraining, dashboard updates, etc.)
    summary = {
        "mlflow_run_id": run_id,
        "mlflow_run_name": mlflow_run_name,
        "drift_detected": drift_summary["drift_detected"],
        "drifted_columns_count": drift_summary["drifted_columns_count"],
        "drifted_columns_share": drift_summary["drifted_columns_share"],
        "baseline_file": f"{baseline_filename}",
        "current_file": f"{current_headers_filename}",
        "monitoring_timestamp": timestamp,
    }
    summary_path = os.path.join(output_dir, "monitoring_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Monitoring processing job completed successfully!")
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
