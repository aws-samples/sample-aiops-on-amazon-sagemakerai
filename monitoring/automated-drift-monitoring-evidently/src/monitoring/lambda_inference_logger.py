"""
Lambda consumer: reads inference logs from SQS and writes to Athena Iceberg table.
"""

import json
import os
import boto3


ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.getenv('ATHENA_OUTPUT_S3', 's3://fraud-detection-data-lake-skoppar-YOUR_ACCOUNT_ID/athena-query-results/')

# Table columns with types for proper NULL casting
COLUMNS = [
    ('inference_id', 'varchar'),
    ('request_timestamp', 'timestamp'),
    ('endpoint_name', 'varchar'),
    ('model_version', 'varchar'),
    ('mlflow_run_id', 'varchar'),
    ('input_features', 'varchar'),
    ('prediction', 'integer'),
    ('probability_fraud', 'double'),
    ('probability_non_fraud', 'double'),
    ('confidence_score', 'double'),
    ('ground_truth', 'integer'),
    ('ground_truth_timestamp', 'timestamp'),
    ('ground_truth_source', 'varchar'),
    ('days_to_ground_truth', 'double'),
    ('inference_latency_ms', 'double'),
    ('model_load_time_ms', 'double'),
    ('preprocessing_time_ms', 'double'),
    ('transaction_id', 'varchar'),
    ('transaction_amount', 'double'),
    ('customer_id', 'varchar'),
    ('is_high_confidence', 'boolean'),
    ('is_low_confidence', 'boolean'),
    ('prediction_bucket', 'varchar'),
    ('request_id', 'varchar'),
    ('response_time', 'timestamp'),
    ('error_message', 'varchar'),
    ('inference_mode', 'varchar'),
]


def sql_val(v, col_type):
    if v is None:
        return f"CAST(NULL AS {col_type})"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Detect ISO timestamp strings
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

    query = f"INSERT INTO {ATHENA_DATABASE}.inference_responses VALUES\n" + ",\n".join(rows)

    athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_S3},
    )

    return {'statusCode': 200, 'body': f'Inserted {len(records)} records'}
