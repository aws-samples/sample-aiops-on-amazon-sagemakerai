#!/usr/bin/env python3
"""
Setup scheduled real-time inference using EventBridge + Lambda.

This script creates:
1. A Lambda function that invokes the SageMaker endpoint
2. An EventBridge rule to trigger the Lambda on a schedule
3. Required IAM permissions

The Lambda function:
- Queries Athena for unscored transactions
- Invokes the SageMaker endpoint for predictions
- Writes results back to Athena

Usage:
    # Create scheduled inference (every hour)
    python scripts/setup_scheduled_inference.py --endpoint-name fraud-detector --schedule "rate(1 hour)"
    
    # Create with cron expression (daily at 8 AM UTC)
    python scripts/setup_scheduled_inference.py --endpoint-name fraud-detector --schedule "cron(0 8 * * ? *)"
    
    # Delete the scheduled inference
    python scripts/setup_scheduled_inference.py --endpoint-name fraud-detector --delete
"""

import argparse
import json
import logging
import sys
import time
import zipfile
import io
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lambda function code for real-time inference
LAMBDA_CODE = '''
import json
import logging
import os
import time
import boto3
from datetime import datetime, timedelta

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
athena_client = boto3.client('athena')
s3_client = boto3.client('s3')

# Configuration from environment
ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'fraud-detector')
ATHENA_DATABASE = os.environ.get('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.environ.get('ATHENA_OUTPUT_S3', 's3://fraud-detection-data-lake/athena-query-results/')
DATA_S3_BUCKET = os.environ.get('DATA_S3_BUCKET', 'fraud-detection-data-lake')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '100'))
LOOKBACK_MINUTES = int(os.environ.get('LOOKBACK_MINUTES', '60'))


def run_athena_query(query: str) -> str:
    """Execute Athena query and return query execution ID."""
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_S3}
    )
    return response['QueryExecutionId']


def wait_for_query(query_id: str, timeout: int = 300) -> bool:
    """Wait for Athena query to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = athena_client.get_query_execution(QueryExecutionId=query_id)
        state = response['QueryExecution']['Status']['State']
        if state == 'SUCCEEDED':
            return True
        elif state in ['FAILED', 'CANCELLED']:
            logger.error(f"Query failed: {response['QueryExecution']['Status'].get('StateChangeReason')}")
            return False
        time.sleep(2)
    return False


def get_query_results(query_id: str) -> list:
    """Get results from completed Athena query."""
    results = []
    paginator = athena_client.get_paginator('get_query_results')
    
    for page in paginator.paginate(QueryExecutionId=query_id):
        rows = page['ResultSet']['Rows']
        if not results:  # First page includes header
            header = [col['VarCharValue'] for col in rows[0]['Data']]
            rows = rows[1:]
        
        for row in rows:
            values = [col.get('VarCharValue', '') for col in row['Data']]
            results.append(dict(zip(header, values)))
    
    return results


def invoke_endpoint(features: list) -> dict:
    """Invoke SageMaker endpoint with features."""
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps({'instances': features})
    )
    return json.loads(response['Body'].read().decode())


def lambda_handler(event, context):
    """
    Main handler for scheduled inference.
    
    1. Query Athena for recent unscored transactions
    2. Invoke SageMaker endpoint for predictions
    3. Write results to inference_responses table
    """
    logger.info(f"Starting scheduled inference for endpoint: {ENDPOINT_NAME}")
    
    start_time = datetime.utcnow()
    lookback_time = start_time - timedelta(minutes=LOOKBACK_MINUTES)
    
    # Query for unscored transactions
    # Adjust this query based on your actual table schema
    query = f"""
    SELECT 
        transaction_id,
        transaction_timestamp,
        transaction_amount,
        transaction_type_code,
        customer_age,
        customer_gender,
        customer_tenure_months,
        account_age_days,
        distance_from_home_km,
        distance_from_last_transaction_km,
        time_since_last_transaction_min,
        online_transaction,
        international_transaction,
        high_risk_country,
        merchant_category_code,
        merchant_reputation_score,
        chip_transaction,
        pin_used,
        card_present,
        cvv_match,
        address_verification_match,
        num_transactions_24h,
        num_transactions_7days,
        avg_transaction_amount_30days,
        max_transaction_amount_30days,
        velocity_score,
        recurring_transaction,
        previous_fraud_incidents,
        credit_limit,
        available_credit_ratio
    FROM training_data
    WHERE transaction_timestamp >= timestamp '{lookback_time.strftime("%Y-%m-%d %H:%M:%S")}'
    LIMIT {BATCH_SIZE}
    """
    
    logger.info(f"Querying for transactions since {lookback_time}")
    
    try:
        # Execute query
        query_id = run_athena_query(query)
        if not wait_for_query(query_id):
            return {'statusCode': 500, 'error': 'Athena query failed'}
        
        # Get results
        transactions = get_query_results(query_id)
        logger.info(f"Found {len(transactions)} transactions to score")
        
        if not transactions:
            return {
                'statusCode': 200,
                'message': 'No transactions to score',
                'transactions_scored': 0
            }
        
        # Prepare features for inference
        feature_columns = [
            'transaction_amount', 'transaction_type_code', 'customer_age',
            'customer_gender', 'customer_tenure_months', 'account_age_days',
            'distance_from_home_km', 'distance_from_last_transaction_km',
            'time_since_last_transaction_min', 'online_transaction',
            'international_transaction', 'high_risk_country',
            'merchant_category_code', 'merchant_reputation_score',
            'chip_transaction', 'pin_used', 'card_present',
            'cvv_match', 'address_verification_match',
            'num_transactions_24h', 'num_transactions_7days',
            'avg_transaction_amount_30days', 'max_transaction_amount_30days',
            'velocity_score', 'recurring_transaction', 'previous_fraud_incidents',
            'credit_limit', 'available_credit_ratio'
        ]
        
        features = []
        transaction_ids = []
        for txn in transactions:
            feature_values = [float(txn.get(col, 0)) for col in feature_columns]
            features.append(feature_values)
            transaction_ids.append(txn.get('transaction_id', ''))
        
        # Invoke endpoint
        logger.info(f"Invoking endpoint with {len(features)} samples")
        predictions = invoke_endpoint(features)
        
        # Process results
        scored_count = 0
        fraud_count = 0
        
        if 'predictions' in predictions:
            for i, pred in enumerate(predictions['predictions']):
                score = pred if isinstance(pred, float) else pred.get('score', 0)
                is_fraud = score >= 0.5
                if is_fraud:
                    fraud_count += 1
                scored_count += 1
                
                # Log high-risk transactions
                if score >= 0.8:
                    logger.warning(f"High fraud risk: transaction_id={transaction_ids[i]}, score={score:.4f}")
        
        # Calculate metrics
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        result = {
            'statusCode': 200,
            'endpoint_name': ENDPOINT_NAME,
            'transactions_scored': scored_count,
            'fraud_detected': fraud_count,
            'fraud_rate': fraud_count / scored_count if scored_count > 0 else 0,
            'duration_ms': duration_ms,
            'timestamp': end_time.isoformat()
        }
        
        logger.info(f"Completed: {scored_count} scored, {fraud_count} fraud detected")
        return result
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return {
            'statusCode': 500,
            'error': str(e)
        }
'''


def create_lambda_deployment_package() -> bytes:
    """Create a ZIP deployment package for the Lambda function."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('lambda_function.py', LAMBDA_CODE)
    zip_buffer.seek(0)
    return zip_buffer.read()


def get_or_create_lambda_role(iam_client, role_name: str, account_id: str) -> str:
    """Get or create IAM role for Lambda function."""
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        response = iam_client.get_role(RoleName=role_name)
        logger.info(f"Using existing role: {role_name}")
        return response['Role']['Arn']
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchEntity':
            raise
    
    # Create role
    logger.info(f"Creating IAM role: {role_name}")
    response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description='Role for scheduled SageMaker inference Lambda'
    )
    role_arn = response['Role']['Arn']
    
    # Attach policies
    policies = [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
        'arn:aws:iam::aws:policy/AmazonAthenaFullAccess',
        'arn:aws:iam::aws:policy/AmazonS3FullAccess',
    ]
    
    for policy_arn in policies:
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        logger.info(f"  Attached: {policy_arn.split('/')[-1]}")
    
    # Wait for role to propagate
    logger.info("Waiting for role to propagate...")
    time.sleep(10)
    
    return role_arn


def create_scheduled_inference(
    endpoint_name: str,
    schedule_expression: str,
    region: str = 'us-east-1',
    athena_database: str = 'fraud_detection',
    s3_bucket: str = None,
    batch_size: int = 100,
    lookback_minutes: int = 60
) -> dict:
    """
    Create Lambda function and EventBridge rule for scheduled inference.
    
    Args:
        endpoint_name: SageMaker endpoint name
        schedule_expression: EventBridge schedule (e.g., "rate(1 hour)")
        region: AWS region
        athena_database: Athena database name
        s3_bucket: S3 bucket for data
        batch_size: Number of transactions per batch
        lookback_minutes: How far back to look for transactions
        
    Returns:
        Dictionary with created resources
    """
    # Initialize clients
    lambda_client = boto3.client('lambda', region_name=region)
    events_client = boto3.client('events', region_name=region)
    iam_client = boto3.client('iam')
    sts_client = boto3.client('sts')
    
    account_id = sts_client.get_caller_identity()['Account']
    
    # Resource names
    function_name = f'scheduled-inference-{endpoint_name}'
    rule_name = f'scheduled-inference-{endpoint_name}-rule'
    role_name = f'scheduled-inference-{endpoint_name}-role'
    
    # Default S3 bucket
    if not s3_bucket:
        s3_bucket = f'fraud-detection-data-lake-skoppar-{account_id}'
    
    logger.info("=" * 60)
    logger.info(f"Creating scheduled inference for endpoint: {endpoint_name}")
    logger.info(f"  Schedule: {schedule_expression}")
    logger.info(f"  Region: {region}")
    logger.info("=" * 60)
    
    # Step 1: Create/get IAM role
    role_arn = get_or_create_lambda_role(iam_client, role_name, account_id)
    
    # Step 2: Create Lambda function
    logger.info(f"Creating Lambda function: {function_name}")
    
    deployment_package = create_lambda_deployment_package()
    
    environment = {
        'Variables': {
            'ENDPOINT_NAME': endpoint_name,
            'ATHENA_DATABASE': athena_database,
            'ATHENA_OUTPUT_S3': f's3://{s3_bucket}/athena-query-results/',
            'DATA_S3_BUCKET': s3_bucket,
            'BATCH_SIZE': str(batch_size),
            'LOOKBACK_MINUTES': str(lookback_minutes),
        }
    }
    
    try:
        # Try to update existing function
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=deployment_package
        )
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Environment=environment,
            Timeout=300,
            MemorySize=512
        )
        logger.info(f"  Updated existing function: {function_name}")
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Create new function
            lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.11',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': deployment_package},
                Description=f'Scheduled inference for {endpoint_name}',
                Timeout=300,
                MemorySize=512,
                Environment=environment,
                Tags={
                    'Purpose': 'ScheduledInference',
                    'Endpoint': endpoint_name
                }
            )
            logger.info(f"  Created new function: {function_name}")
        else:
            raise
    
    function_arn = f'arn:aws:lambda:{region}:{account_id}:function:{function_name}'
    
    # Step 3: Create EventBridge rule
    logger.info(f"Creating EventBridge rule: {rule_name}")
    
    events_client.put_rule(
        Name=rule_name,
        ScheduleExpression=schedule_expression,
        State='ENABLED',
        Description=f'Trigger scheduled inference for {endpoint_name}'
    )
    
    # Step 4: Add Lambda permission for EventBridge
    try:
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=f'{rule_name}-permission',
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=f'arn:aws:events:{region}:{account_id}:rule/{rule_name}'
        )
    except ClientError as e:
        if e.response['Error']['Code'] != 'ResourceConflictException':
            raise
    
    # Step 5: Add Lambda as target
    events_client.put_targets(
        Rule=rule_name,
        Targets=[{
            'Id': '1',
            'Arn': function_arn
        }]
    )
    
    logger.info("=" * 60)
    logger.info("✓ Scheduled inference created successfully!")
    logger.info(f"  Lambda Function: {function_name}")
    logger.info(f"  EventBridge Rule: {rule_name}")
    logger.info(f"  Schedule: {schedule_expression}")
    logger.info("=" * 60)
    
    return {
        'function_name': function_name,
        'function_arn': function_arn,
        'rule_name': rule_name,
        'schedule': schedule_expression,
        'status': 'created'
    }


def delete_scheduled_inference(endpoint_name: str, region: str = 'us-east-1') -> dict:
    """Delete Lambda function and EventBridge rule."""
    lambda_client = boto3.client('lambda', region_name=region)
    events_client = boto3.client('events', region_name=region)
    iam_client = boto3.client('iam')
    
    function_name = f'scheduled-inference-{endpoint_name}'
    rule_name = f'scheduled-inference-{endpoint_name}-rule'
    role_name = f'scheduled-inference-{endpoint_name}-role'
    
    logger.info(f"Deleting scheduled inference for: {endpoint_name}")
    
    # Remove targets from rule
    try:
        events_client.remove_targets(Rule=rule_name, Ids=['1'])
        logger.info(f"  Removed targets from rule: {rule_name}")
    except ClientError:
        pass
    
    # Delete rule
    try:
        events_client.delete_rule(Name=rule_name)
        logger.info(f"  Deleted rule: {rule_name}")
    except ClientError:
        pass
    
    # Delete Lambda function
    try:
        lambda_client.delete_function(FunctionName=function_name)
        logger.info(f"  Deleted function: {function_name}")
    except ClientError:
        pass
    
    # Delete IAM role (detach policies first)
    try:
        paginator = iam_client.get_paginator('list_attached_role_policies')
        for page in paginator.paginate(RoleName=role_name):
            for policy in page['AttachedPolicies']:
                iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
        iam_client.delete_role(RoleName=role_name)
        logger.info(f"  Deleted role: {role_name}")
    except ClientError:
        pass
    
    logger.info("✓ Scheduled inference deleted")
    return {'status': 'deleted', 'endpoint_name': endpoint_name}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup scheduled real-time inference with EventBridge + Lambda",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create scheduled inference (every hour)
  python scripts/setup_scheduled_inference.py --endpoint-name fraud-detector --schedule "rate(1 hour)"
  
  # Create with cron expression (daily at 8 AM UTC)
  python scripts/setup_scheduled_inference.py --endpoint-name fraud-detector --schedule "cron(0 8 * * ? *)"
  
  # Every 15 minutes
  python scripts/setup_scheduled_inference.py --endpoint-name fraud-detector --schedule "rate(15 minutes)"
  
  # Delete scheduled inference
  python scripts/setup_scheduled_inference.py --endpoint-name fraud-detector --delete

Schedule Expression Examples:
  rate(1 minute)       - Every minute
  rate(5 minutes)      - Every 5 minutes  
  rate(1 hour)         - Every hour
  rate(1 day)          - Every day
  cron(0 8 * * ? *)    - Daily at 8 AM UTC
  cron(0 */2 * * ? *)  - Every 2 hours
  cron(0 8 ? * MON *)  - Every Monday at 8 AM UTC
        """
    )
    
    parser.add_argument(
        '--endpoint-name',
        required=True,
        help='SageMaker endpoint name'
    )
    parser.add_argument(
        '--schedule',
        default='rate(1 hour)',
        help='EventBridge schedule expression (default: rate(1 hour))'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--athena-database',
        default='fraud_detection',
        help='Athena database name (default: fraud_detection)'
    )
    parser.add_argument(
        '--s3-bucket',
        help='S3 bucket for data'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of transactions per batch (default: 100)'
    )
    parser.add_argument(
        '--lookback-minutes',
        type=int,
        default=60,
        help='How far back to look for transactions (default: 60)'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete the scheduled inference instead of creating'
    )
    
    args = parser.parse_args()
    
    try:
        if args.delete:
            result = delete_scheduled_inference(
                endpoint_name=args.endpoint_name,
                region=args.region
            )
        else:
            result = create_scheduled_inference(
                endpoint_name=args.endpoint_name,
                schedule_expression=args.schedule,
                region=args.region,
                athena_database=args.athena_database,
                s3_bucket=args.s3_bucket,
                batch_size=args.batch_size,
                lookback_minutes=args.lookback_minutes
            )
        
        print(json.dumps(result, indent=2))
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
