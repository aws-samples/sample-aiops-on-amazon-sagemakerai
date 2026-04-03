#!/usr/bin/env python3
"""
Deploy monitoring results writer Lambda function.

This Lambda consumes messages from SQS and writes monitoring results
to the monitoring_responses Iceberg table in Athena.
"""

import boto3
import json
import zipfile
import io
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def deploy_monitoring_writer(region='us-east-1'):
    """Deploy the monitoring writer Lambda."""

    # Load .env if available
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  Deploying Monitoring Results Writer Lambda                       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("")

    # Get AWS info
    sts = boto3.client('sts', region_name=region)
    account_id = sts.get_caller_identity()['Account']

    # Configuration (from .env with defaults)
    lambda_name = os.getenv('MONITORING_WRITER_LAMBDA_NAME', 'fraud-monitoring-results-writer')
    role_name = lambda_name + '-role'
    queue_name = os.getenv('MONITORING_SQS_QUEUE_NAME', 'fraud-monitoring-results')
    database = os.getenv('ATHENA_DATABASE', 'fraud_detection')
    table = os.getenv('MONITORING_TABLE_NAME', 'monitoring_responses')
    bucket = os.getenv('DATA_S3_BUCKET', f'fraud-detection-data-lake-skoppar-{account_id}')
    output_location = f's3://{bucket}/athena-query-results/'

    print(f"  Region: {region}")
    print(f"  Account: {account_id}")
    print(f"  Lambda: {lambda_name}")
    print(f"  Queue: {queue_name}")
    print("")

    # Clients
    iam = boto3.client('iam', region_name=region)
    sqs = boto3.client('sqs', region_name=region)
    lambda_client = boto3.client('lambda', region_name=region)

    # Step 1: Create SQS queue
    print("[1/5] Creating SQS queue...")
    try:
        queue_url = sqs.create_queue(
            QueueName=queue_name,
            Attributes={
                'VisibilityTimeout': '300',  # 5 minutes
                'MessageRetentionPeriod': '1209600',  # 14 days
            }
        )['QueueUrl']
        print(f"  ✓ Queue created: {queue_url}")
    except sqs.exceptions.QueueNameExists:
        queue_url = sqs.get_queue_url(QueueName=queue_name)['QueueUrl']
        print(f"  ✓ Queue exists: {queue_url}")

    queue_arn = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['QueueArn']
    )['Attributes']['QueueArn']

    # Step 2: Create IAM role
    print("")
    print("[2/5] Creating IAM role...")
    trust_policy = {
        'Version': '2012-10-17',
        'Statement': [{
            'Effect': 'Allow',
            'Principal': {'Service': 'lambda.amazonaws.com'},
            'Action': 'sts:AssumeRole'
        }]
    }

    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Monitoring results writer Lambda role'
        )
        print("  ✓ Role created")
    except iam.exceptions.EntityAlreadyExistsException:
        print("  ✓ Role exists")

    role_arn = f'arn:aws:iam::{account_id}:role/{role_name}'

    # Attach managed policies
    for policy_arn in [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/service-role/AWSLambdaSQSQueueExecutionRole',
        'arn:aws:iam::aws:policy/AmazonAthenaFullAccess',
    ]:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

    print("  ✓ Policies attached")
    print("  Waiting 10s for role propagation...")
    time.sleep(10)

    # Step 3: Create Lambda deployment package
    print("")
    print("[3/5] Creating Lambda package...")

    # Lambda code
    lambda_code = '''
import json
import boto3
import time
import os
from datetime import datetime

athena = boto3.client('athena')

ATHENA_DATABASE = os.environ['ATHENA_DATABASE']
ATHENA_TABLE = os.environ['ATHENA_TABLE']
ATHENA_OUTPUT = os.environ['ATHENA_OUTPUT']

def lambda_handler(event, context):
    """Process SQS messages and write to Athena."""
    print(f"Processing {len(event['Records'])} messages")

    for record in event['Records']:
        try:
            body = json.loads(record['body'])
            write_to_athena(body)
        except Exception as e:
            print(f"Error processing message: {e}")
            raise

    return {'statusCode': 200, 'processed': len(event['Records'])}

def write_to_athena(data):
    """Write monitoring result to Athena Iceberg table."""
    # Build INSERT query
    columns = [
        'timestamp', 'run_id', 'detection_engine',
        'data_drift_detected', 'data_drift_score',
        'drifted_columns_count', 'drifted_columns_share',
        'model_drift_detected', 'baseline_roc_auc', 'current_roc_auc',
        'degradation', 'degradation_pct',
        'accuracy', 'precision_score', 'recall', 'f1_score',
        'sample_size', 'alert_sent', 'details'
    ]

    values = []
    for col in columns:
        val = data.get(col)
        if val is None:
            values.append('NULL')
        elif isinstance(val, bool):
            values.append('TRUE' if val else 'FALSE')
        elif isinstance(val, (int, float)):
            values.append(str(val))
        else:
            # Escape single quotes in strings
            val_str = str(val).replace("'", "''")
            values.append(f"'{val_str}'")

    query = f"""
    INSERT INTO {ATHENA_DATABASE}.{ATHENA_TABLE}
    VALUES ({', '.join(values)})
    """

    # Execute query
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT}
    )

    execution_id = response['QueryExecutionId']
    print(f"Athena query started: {execution_id}")

    # Wait for completion
    while True:
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(0.5)

    if state == 'SUCCEEDED':
        print(f"✓ Written to {ATHENA_DATABASE}.{ATHENA_TABLE}")
    else:
        reason = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
        raise Exception(f"Athena query failed: {reason}")
'''

    # Create ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('lambda_function.py', lambda_code)
    zip_bytes = buf.getvalue()

    print(f"  ✓ Package created: {len(zip_bytes) / 1024:.1f} KB")

    # Step 4: Create/Update Lambda function
    print("")
    print("[4/5] Deploying Lambda function...")

    env_vars = {
        'ATHENA_DATABASE': database,
        'ATHENA_TABLE': table,
        'ATHENA_OUTPUT': output_location
    }

    try:
        lambda_client.get_function(FunctionName=lambda_name)
        # Update existing
        lambda_client.update_function_code(
            FunctionName=lambda_name,
            ZipFile=zip_bytes
        )
        lambda_client.get_waiter('function_updated_v2').wait(FunctionName=lambda_name)
        lambda_client.update_function_configuration(
            FunctionName=lambda_name,
            Timeout=60,
            MemorySize=256,
            Environment={'Variables': env_vars}
        )
        print("  ✓ Lambda updated")
    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new
        lambda_client.create_function(
            FunctionName=lambda_name,
            Runtime='python3.11',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_bytes},
            Timeout=60,
            MemorySize=256,
            Environment={'Variables': env_vars},
            Description='Write monitoring results to Athena'
        )
        print("  ✓ Lambda created")

    lambda_arn = f'arn:aws:lambda:{region}:{account_id}:function:{lambda_name}'

    # Step 5: Configure SQS trigger
    print("")
    print("[5/5] Configuring SQS trigger...")

    try:
        lambda_client.create_event_source_mapping(
            EventSourceArn=queue_arn,
            FunctionName=lambda_name,
            BatchSize=10,
            MaximumBatchingWindowInSeconds=5
        )
        print("  ✓ SQS trigger configured")
    except lambda_client.exceptions.ResourceConflictException:
        print("  ✓ SQS trigger exists")

    print("")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ MONITORING WRITER DEPLOYED                                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"  Lambda: {lambda_arn}")
    print(f"  Queue: {queue_url}")
    print(f"  Target: {database}.{table}")
    print("")
    print("Test: Send message to SQS queue with monitoring data")
    print("")

    return queue_url

if __name__ == '__main__':
    region = sys.argv[1] if len(sys.argv) > 1 else 'us-east-1'
    deploy_monitoring_writer(region)
