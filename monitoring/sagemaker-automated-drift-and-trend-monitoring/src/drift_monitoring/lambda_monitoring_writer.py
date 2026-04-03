"""
Lambda consumer: reads monitoring results from SQS and writes to Athena Iceberg table.

Same pattern as lambda_inference_logger.py but for the monitoring_responses table.
"""

import json
import os
import boto3


ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.getenv('ATHENA_OUTPUT_S3', 's3://fraud-detection-data-lake/athena-query-results/')

# Column definitions with types for proper NULL casting
COLUMNS = [
    ('monitoring_run_id', 'varchar'),
    ('monitoring_timestamp', 'timestamp'),
    ('endpoint_name', 'varchar'),
    ('model_version', 'varchar'),
    ('data_drift_detected', 'boolean'),
    ('drifted_columns_count', 'integer'),
    ('drifted_columns_share', 'double'),
    ('features_analyzed', 'integer'),
    ('data_sample_size', 'integer'),
    ('model_drift_detected', 'boolean'),
    ('baseline_roc_auc', 'double'),
    ('current_roc_auc', 'double'),
    ('roc_auc_degradation', 'double'),
    ('roc_auc_degradation_pct', 'double'),
    ('accuracy', 'double'),
    ('precision', 'double'),
    ('recall', 'double'),
    ('f1_score', 'double'),
    ('model_sample_size', 'integer'),
    ('per_feature_drift_scores', 'varchar'),
    ('evidently_report_s3_path', 'varchar'),
    ('mlflow_run_id', 'varchar'),
    ('alert_sent', 'boolean'),
    ('detection_engine', 'varchar'),
    ('created_at', 'timestamp'),
]


def sql_val(v, col_type):
    if v is None:
        return f"CAST(NULL AS {col_type})"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if col_type == 'timestamp' and len(s) >= 19 and s[4] == '-' and s[7] == '-':
        return f"TIMESTAMP '{s.replace('T', ' ')}'"
    return f"'{s.replace(chr(39), chr(39)+chr(39))}'"


def lambda_handler(event, context):
    athena = boto3.client('athena')

    records = []
    for sqs_record in event.get('Records', []):
        body = json.loads(sqs_record['body'])
        records.append(body)

    if not records:
        return {'statusCode': 200, 'body': 'No records'}

    rows = []
    for r in records:
        row = ", ".join(
            sql_val(r.get(col_name), col_type)
            for col_name, col_type in COLUMNS
        )
        rows.append(f"({row})")

    query = f"INSERT INTO {ATHENA_DATABASE}.monitoring_responses VALUES\n" + ",\n".join(rows)

    athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_S3},
    )

    return {'statusCode': 200, 'body': f'Inserted {len(records)} monitoring records'}
