import json
import os
import base64
import boto3
import pandas as pd
import mlflow
from datetime import datetime
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset

s3_client = boto3.client('s3')

BUCKET = os.environ['BUCKET']
BASELINE_KEY = os.environ['BASELINE_KEY']
CAPTURE_PREFIX = os.environ['CAPTURE_PREFIX']
MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
MLFLOW_EXPERIMENT = os.environ['MLFLOW_EXPERIMENT']
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

DRIFT_THRESHOLD = 0.1  # Only log per-column drift metrics above this threshold


def parse_capture_files(bucket, prefix):
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    records = []
    for obj in resp.get('Contents', []):
        body = s3_client.get_object(Bucket=bucket, Key=obj['Key'])['Body'].read().decode('utf-8')
        for line in body.strip().split('\n'):
            rec = json.loads(line)
            input_data = rec.get('captureData', {}).get('endpointInput', {})
            raw = input_data.get('data', '')
            if input_data.get('encoding') == 'BASE64':
                raw = base64.b64decode(raw).decode('utf-8')
            records.append(raw.strip().split(','))
    return records


def log_drift_metrics(report_dict):
    """Extract and log individual drift metrics from the Evidently report dict."""
    metrics = report_dict.get('metrics', []) or []
    drifted_columns_logged = False

    for metric in metrics:
        metric_name = metric.get('metric_name', '')
        value = metric.get('value')

        # Log DriftedColumnsCount as count and share
        if metric_name == 'DriftedColumnsCount':
            drifted_columns_logged = True
            if isinstance(value, dict):
                count = float(value.get('count', 0))
                share = float(value.get('share', 0))
            else:
                count = float(value) if value is not None else 0.0
                share = 0.0
            mlflow.log_metric('DriftedColumnsCount.count', count)
            mlflow.log_metric('DriftedColumnsCount.share', share)
            continue

        # Log per-column ValueDrift above threshold
        if metric_name == 'ValueDrift':
            column = metric.get('column', 'unknown')
            try:
                numeric_val = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_val > DRIFT_THRESHOLD:
                mlflow.log_metric(f'ValueDrift:{column}', numeric_val)

    if not drifted_columns_logged:
        mlflow.log_metric('DriftedColumnsCount.count', 0.0)
        mlflow.log_metric('DriftedColumnsCount.share', 0.0)


def handler(event, context):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    baseline_obj = s3_client.get_object(Bucket=BUCKET, Key=BASELINE_KEY)
    reference_data = pd.read_csv(baseline_obj['Body'])

    raw_records = parse_capture_files(BUCKET, CAPTURE_PREFIX)
    production_data = pd.DataFrame(raw_records, columns=reference_data.columns).astype(float)

    data_definition = DataDefinition()
    reference_dataset = Dataset.from_pandas(reference_data, data_definition=data_definition)
    production_dataset = Dataset.from_pandas(production_data, data_definition=data_definition)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with mlflow.start_run(run_name=f'data_drift_{timestamp}'):
        mlflow.log_params({
            'endpoint_name': ENDPOINT_NAME,
            'reference_data_size': len(reference_data),
            'current_data_size': len(production_data),
            'monitoring_timestamp': datetime.now().isoformat(),
            'data_capture_s3_uri': f's3://{BUCKET}/{CAPTURE_PREFIX}',
        })

        # Data drift report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_snapshot = drift_report.run(reference_data=reference_dataset, current_data=production_dataset)

        drift_html = f'/tmp/data_drift_{timestamp}.html'
        drift_json = f'/tmp/data_drift_{timestamp}.json'
        drift_snapshot.save_html(drift_html)
        drift_snapshot.save_json(drift_json)
        mlflow.log_artifact(drift_html, 'evidently_report_data_drift_html')
        mlflow.log_artifact(drift_json, 'evidently_report_data_drift_json')

        # Extract and log individual drift metrics
        drift_results = drift_snapshot.dict()
        log_drift_metrics(drift_results)

        # Data summary report
        summary_report = Report(metrics=[DataSummaryPreset()])
        summary_snapshot = summary_report.run(reference_data=reference_dataset, current_data=production_dataset)

        summary_html = f'/tmp/data_summary_{timestamp}.html'
        summary_snapshot.save_html(summary_html)
        mlflow.log_artifact(summary_html, 'evidently_report_data_quality')

    return {'statusCode': 200, 'body': 'Data drift report complete'}
