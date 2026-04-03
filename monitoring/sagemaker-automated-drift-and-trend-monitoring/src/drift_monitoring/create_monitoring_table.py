#!/usr/bin/env python3
"""
Create monitoring_responses Iceberg table in Athena.

This table stores monitoring metrics and drift detection results
for long-term trend analysis and governance dashboards.
"""

import boto3
import time
import sys

def create_monitoring_table(region='us-east-1'):
    """Create monitoring_responses Iceberg table."""

    # Get AWS account info
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()['Account']

    # Configuration
    database = 'fraud_detection'
    table = 'monitoring_responses'
    bucket = f'fraud-detection-data-lake-skoppar-{account_id}'
    output_location = f's3://{bucket}/athena-query-results/'
    table_location = f's3://{bucket}/monitoring-iceberg/'

    print(f"Creating monitoring table in {database}.{table}")
    print(f"  Region: {region}")
    print(f"  Account: {account_id}")
    print(f"  Location: {table_location}")
    print("")

    # Create Athena client
    athena = boto3.client('athena', region_name=region)

    # Drop existing table if needed
    print("[1/2] Dropping existing table (if any)...")
    drop_query = f"DROP TABLE IF EXISTS {database}.{table}"

    response = athena.start_query_execution(
        QueryString=drop_query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output_location}
    )

    execution_id = response['QueryExecutionId']

    # Wait for drop to complete
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)

    if state == 'SUCCEEDED':
        print("  ✓ Old table dropped")
    else:
        print(f"  Note: Drop query state={state}")

    # Create new Iceberg table
    print("")
    print("[2/2] Creating new Iceberg table...")

    create_query = f"""
CREATE TABLE {database}.{table} (
    timestamp STRING,
    run_id STRING,
    detection_engine STRING,
    data_drift_detected BOOLEAN,
    data_drift_score DOUBLE,
    drifted_columns_count INT,
    drifted_columns_share DOUBLE,
    model_drift_detected BOOLEAN,
    baseline_roc_auc DOUBLE,
    current_roc_auc DOUBLE,
    degradation DOUBLE,
    degradation_pct DOUBLE,
    accuracy DOUBLE,
    precision_score DOUBLE,
    recall DOUBLE,
    f1_score DOUBLE,
    sample_size INT,
    alert_sent BOOLEAN,
    details STRING
)
LOCATION '{table_location}'
TBLPROPERTIES (
  'table_type' = 'ICEBERG',
  'format' = 'parquet',
  'write_compression' = 'snappy',
  'optimize_rewrite_delete_file_threshold' = '10'
)
"""

    response = athena.start_query_execution(
        QueryString=create_query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output_location}
    )

    execution_id = response['QueryExecutionId']

    # Wait for creation to complete
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)

    if state == 'SUCCEEDED':
        print("  ✓ Table created successfully")
        print("")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║  ✅ MONITORING TABLE CREATED                                       ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print("")
        print(f"  Database: {database}")
        print(f"  Table: {table}")
        print(f"  Type: Iceberg (ACID-compliant)")
        print(f"  Location: {table_location}")
        print("")
        print("Next steps:")
        print(f"  1. Grant Lake Formation permissions")
        print(f"  2. Query: SELECT * FROM {database}.{table} LIMIT 10")
        print("")
        return True
    else:
        reason = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
        print(f"  ❌ Table creation failed: {reason}")
        return False

if __name__ == '__main__':
    region = sys.argv[1] if len(sys.argv) > 1 else 'us-east-1'
    success = create_monitoring_table(region)
    sys.exit(0 if success else 1)
