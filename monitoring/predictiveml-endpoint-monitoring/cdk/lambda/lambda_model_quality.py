import json
import os
import base64
import boto3
import pandas as pd
import mlflow
from datetime import datetime
from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import ClassificationPreset

s3_client = boto3.client('s3')
sns_client = boto3.client('sns')

BUCKET = os.environ['BUCKET']
CAPTURE_PREFIX = os.environ['CAPTURE_PREFIX']
GROUND_TRUTH_KEY = os.environ['GROUND_TRUTH_KEY']
MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
MLFLOW_EXPERIMENT = os.environ['MLFLOW_EXPERIMENT']
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
FEATURE_COLUMNS = os.environ['FEATURE_COLUMNS'].split(',')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')
QUALITY_THRESHOLDS = {
    'F1Score': float(os.environ.get('THRESHOLD_F1', '0.7')),
    'Accuracy': float(os.environ.get('THRESHOLD_ACCURACY', '0.8')),
    'RocAuc': float(os.environ.get('THRESHOLD_ROC_AUC', '0.75')),
}


def parse_capture_files(bucket, prefix):
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    records = []
    for obj in resp.get('Contents', []):
        body = s3_client.get_object(Bucket=bucket, Key=obj['Key'])['Body'].read().decode('utf-8')
        for line in body.strip().split('\n'):
            rec = json.loads(line)
            inference_id = rec.get('eventMetadata', {}).get('inferenceId', '')
            input_data = rec.get('captureData', {}).get('endpointInput', {})
            output_data = rec.get('captureData', {}).get('endpointOutput', {})
            raw_in = input_data.get('data', '')
            if input_data.get('encoding') == 'BASE64':
                raw_in = base64.b64decode(raw_in).decode('utf-8')
            raw_out = output_data.get('data', '')
            if output_data.get('encoding') == 'BASE64':
                raw_out = base64.b64decode(raw_out).decode('utf-8')
            records.append({
                'inference_id': inference_id,
                'prediction_proba': float(raw_out.strip()),
                **dict(zip(FEATURE_COLUMNS, raw_in.strip().split(','))),
            })
    return records


def log_classification_metrics(report_dict):
    """Extract and log individual classification metrics from the Evidently report dict."""
    for metric in report_dict.get('metrics', []):
        metric_name = metric.get('metric_name', '').split('(')[0].strip()
        value = metric.get('value')

        if isinstance(value, dict):
            # Per-label metrics like F1ByLabel, PrecisionByLabel
            for label, val in value.items():
                try:
                    mlflow.log_metric(f'{metric_name}_label_{label}', float(val))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                mlflow.log_metric(metric_name, float(value))
            except (TypeError, ValueError):
                continue


def check_thresholds(report_dict, thresholds):
    breached = {}
    for metric in report_dict.get('metrics', []):
        name = metric.get('metric_name', '').split('(')[0].strip()
        if name in thresholds:
            try:
                val = float(metric.get('value'))
            except (TypeError, ValueError):
                continue
            if val < thresholds[name]:
                breached[name] = val
    return breached


def send_alert(topic_arn, breached, run_name, experiment_name, endpoint_name):
    lines = '\n'.join(f'  - {m}: {v:.4f} (threshold={QUALITY_THRESHOLDS[m]:.4f})' for m, v in breached.items())
    message = (
        f'MODEL QUALITY DEGRADATION DETECTED\n'
        f'{"=" * 50}\n\n'
        f'MLflow Experiment : {experiment_name}\n'
        f'MLflow Run        : {run_name}\n'
        f'Endpoint          : {endpoint_name}\n\n'
        f'Metrics below threshold:\n{lines}\n\n'
        f'Review the full Evidently report in the MLflow artifacts.\n'
    )
    sns_client.publish(TopicArn=topic_arn, Subject=f'Model Quality Alert - {experiment_name}', Message=message)


def handler(event, context):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    raw_records = parse_capture_files(BUCKET, CAPTURE_PREFIX)
    captured_df = pd.DataFrame(raw_records)
    captured_df['prediction'] = (captured_df['prediction_proba'] > 0.5).astype(int)

    gt_obj = s3_client.get_object(Bucket=BUCKET, Key=GROUND_TRUTH_KEY)
    gt_df = pd.read_csv(gt_obj['Body'])
    eval_data = captured_df.merge(gt_df, on='inference_id', how='inner')

    data_definition = DataDefinition(
        classification=[BinaryClassification(target='target', prediction_labels='prediction', pos_label=1)]
    )
    eval_dataset = Dataset.from_pandas(eval_data, data_definition=data_definition)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'model_quality_{timestamp}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'endpoint_name': ENDPOINT_NAME,
            'eval_data_size': len(eval_data),
            'captured_data_size': len(captured_df),
            'ground_truth_size': len(gt_df),
            'monitoring_timestamp': datetime.now().isoformat(),
            'data_capture_s3_uri': f's3://{BUCKET}/{CAPTURE_PREFIX}',
        })

        report = Report(metrics=[ClassificationPreset()])
        snapshot = report.run(reference_data=None, current_data=eval_dataset)

        # Save both HTML and JSON artifacts
        html_path = f'/tmp/classification_report_{timestamp}.html'
        json_path = f'/tmp/classification_report_{timestamp}.json'
        snapshot.save_html(html_path)
        snapshot.save_json(json_path)
        mlflow.log_artifact(html_path, 'evidently_classification_report_html')
        mlflow.log_artifact(json_path, 'evidently_classification_report_json')

        # Extract and log individual classification metrics
        report_dict = snapshot.dict()
        log_classification_metrics(report_dict)

        # Check thresholds and alert
        breached = check_thresholds(report_dict, QUALITY_THRESHOLDS)
        if breached and SNS_TOPIC_ARN:
            send_alert(SNS_TOPIC_ARN, breached, run_name, MLFLOW_EXPERIMENT, ENDPOINT_NAME)

    return {'statusCode': 200, 'body': f'Model quality report complete. Breached metrics: {list(breached.keys())}'}
