#!/usr/bin/env python3
"""
Setup scheduled batch transform using EventBridge + Lambda + SageMaker.

This script creates:
1. A Lambda function that triggers SageMaker batch transform jobs
2. An EventBridge rule to trigger the Lambda on a schedule
3. Required IAM permissions

The batch transform:
- Queries Athena for data to score
- Exports to S3
- Runs SageMaker batch transform
- Writes results back to Athena

Usage:
    # Create scheduled batch transform (daily at 2 AM UTC)
    python scripts/setup_scheduled_batch_transform.py \\
        --model-name sg-xgboost-fraud-detector \\
        --schedule "cron(0 2 * * ? *)"
    
    # Create with rate expression (every 6 hours)
    python scripts/setup_scheduled_batch_transform.py \\
        --model-name sg-xgboost-fraud-detector \\
        --schedule "rate(6 hours)"
    
    # Delete the scheduled batch transform
    python scripts/setup_scheduled_batch_transform.py \\
        --model-name sg-xgboost-fraud-detector \\
        --delete
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

# Lambda function code for batch transform
LAMBDA_CODE = '''
import json
import logging
import os
import time
import boto3
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clients
sagemaker_client = boto3.client('sagemaker')
athena_client = boto3.client('athena')
s3_client = boto3.client('s3')

# Configuration from environment
MODEL_NAME = os.environ.get('MODEL_NAME', 'sg-xgboost-fraud-detector')
ATHENA_DATABASE = os.environ.get('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.environ.get('ATHENA_OUTPUT_S3')
DATA_S3_BUCKET = os.environ.get('DATA_S3_BUCKET')
SAGEMAKER_ROLE = os.environ.get('SAGEMAKER_ROLE')
INSTANCE_TYPE = os.environ.get('INSTANCE_TYPE', 'ml.m5.xlarge')
INSTANCE_COUNT = int(os.environ.get('INSTANCE_COUNT', '1'))
MAX_CONCURRENT = int(os.environ.get('MAX_CONCURRENT', '4'))
INPUT_TABLE = os.environ.get('INPUT_TABLE', 'training_data')
LOOKBACK_HOURS = int(os.environ.get('LOOKBACK_HOURS', '24'))
ROW_LIMIT = int(os.environ.get('ROW_LIMIT', '10000'))


def run_athena_query(query: str) -> str:
    """Execute Athena query and return query execution ID."""
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_OUTPUT_S3}
    )
    return response['QueryExecutionId']


def wait_for_query(query_id: str, timeout: int = 600) -> bool:
    """Wait for Athena query to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = athena_client.get_query_execution(QueryExecutionId=query_id)
        state = response['QueryExecution']['Status']['State']
        if state == 'SUCCEEDED':
            return True
        elif state in ['FAILED', 'CANCELLED']:
            reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
            logger.error(f"Query failed: {reason}")
            return False
        time.sleep(5)
    logger.error("Query timed out")
    return False


def get_latest_model_package(model_package_group: str) -> str:
    """Get the latest approved model package ARN."""
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if response['ModelPackageSummaryList']:
        return response['ModelPackageSummaryList'][0]['ModelPackageArn']
    return None


def create_model_from_package(model_package_arn: str, model_name: str) -> str:
    """Create a SageMaker model from a model package."""
    try:
        sagemaker_client.describe_model(ModelName=model_name)
        logger.info(f"Model {model_name} already exists")
        return model_name
    except ClientError:
        pass
    
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        },
        ExecutionRoleArn=SAGEMAKER_ROLE
    )
    logger.info(f"Created model: {model_name}")
    return model_name


def lambda_handler(event, context):
    """
    Main handler for scheduled batch transform.
    
    1. Query Athena for data to score
    2. Export to S3 in CSV format
    3. Start SageMaker batch transform job
    4. Return job details (monitoring handled separately)
    """
    logger.info(f"Starting scheduled batch transform for model: {MODEL_NAME}")
    
    job_id = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    job_name = f'batch-transform-{MODEL_NAME}-{job_id}'
    
    # Calculate time window
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=LOOKBACK_HOURS)
    
    # S3 paths
    input_prefix = f'batch-transform-input/{job_id}'
    output_prefix = f'batch-transform-output/{job_id}'
    input_s3_uri = f's3://{DATA_S3_BUCKET}/{input_prefix}'
    output_s3_uri = f's3://{DATA_S3_BUCKET}/{output_prefix}'
    
    try:
        # Step 1: Export data from Athena to S3
        logger.info(f"Exporting data from {INPUT_TABLE} to S3...")
        
        # Feature columns for the model
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
        
        # CTAS query to export data
        export_table = f'batch_export_{job_id.replace("-", "_")}'
        export_query = f"""
        CREATE TABLE {export_table}
        WITH (
            format = 'TEXTFILE',
            field_delimiter = ',',
            external_location = '{input_s3_uri}'
        ) AS
        SELECT {', '.join(feature_columns)}
        FROM {INPUT_TABLE}
        WHERE transaction_timestamp >= timestamp '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
          AND transaction_timestamp < timestamp '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
        LIMIT {ROW_LIMIT}
        """
        
        query_id = run_athena_query(export_query)
        if not wait_for_query(query_id):
            return {'statusCode': 500, 'error': 'Failed to export data from Athena'}
        
        # Get row count
        count_query = f"SELECT COUNT(*) as cnt FROM {export_table}"
        count_id = run_athena_query(count_query)
        wait_for_query(count_id)
        
        # Clean up temp table
        drop_query = f"DROP TABLE IF EXISTS {export_table}"
        run_athena_query(drop_query)
        
        logger.info(f"Data exported to: {input_s3_uri}")
        
        # Step 2: Get or create model
        # Try to get latest model package first
        model_package_arn = get_latest_model_package(MODEL_NAME)
        
        if model_package_arn:
            batch_model_name = f'{MODEL_NAME}-batch-{job_id}'
            create_model_from_package(model_package_arn, batch_model_name)
        else:
            # Use existing model
            batch_model_name = MODEL_NAME
            logger.info(f"Using existing model: {batch_model_name}")
        
        # Step 3: Start batch transform job
        logger.info(f"Starting batch transform job: {job_name}")
        
        sagemaker_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=batch_model_name,
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_s3_uri
                    }
                },
                'ContentType': 'text/csv',
                'SplitType': 'Line'
            },
            TransformOutput={
                'S3OutputPath': output_s3_uri,
                'AssembleWith': 'Line'
            },
            TransformResources={
                'InstanceType': INSTANCE_TYPE,
                'InstanceCount': INSTANCE_COUNT
            },
            MaxConcurrentTransforms=MAX_CONCURRENT,
            MaxPayloadInMB=6,
            BatchStrategy='MultiRecord',
            Tags=[
                {'Key': 'Purpose', 'Value': 'ScheduledBatchTransform'},
                {'Key': 'Model', 'Value': MODEL_NAME},
                {'Key': 'JobId', 'Value': job_id}
            ]
        )
        
        logger.info(f"Batch transform job started: {job_name}")
        
        return {
            'statusCode': 200,
            'job_name': job_name,
            'model_name': batch_model_name,
            'input_s3_uri': input_s3_uri,
            'output_s3_uri': output_s3_uri,
            'time_window': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'status': 'InProgress'
        }
        
    except Exception as e:
        logger.error(f"Error starting batch transform: {e}")
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
        Description='Role for scheduled SageMaker batch transform Lambda'
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
    
    # Add inline policy for IAM PassRole
    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": f"arn:aws:iam::{account_id}:role/*",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": "sagemaker.amazonaws.com"
                }
            }
        }]
    }
    
    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName='SageMakerPassRole',
        PolicyDocument=json.dumps(inline_policy)
    )
    
    # Wait for role to propagate
    logger.info("Waiting for role to propagate...")
    time.sleep(10)
    
    return role_arn


def create_scheduled_batch_transform(
    model_name: str,
    schedule_expression: str,
    region: str = 'us-east-1',
    sagemaker_role: str = None,
    athena_database: str = 'fraud_detection',
    s3_bucket: str = None,
    instance_type: str = 'ml.m5.xlarge',
    instance_count: int = 1,
    max_concurrent: int = 4,
    input_table: str = 'training_data',
    lookback_hours: int = 24,
    row_limit: int = 10000
) -> dict:
    """
    Create Lambda function and EventBridge rule for scheduled batch transform.
    
    Args:
        model_name: SageMaker model or model package group name
        schedule_expression: EventBridge schedule (e.g., "cron(0 2 * * ? *)")
        region: AWS region
        sagemaker_role: SageMaker execution role ARN
        athena_database: Athena database name
        s3_bucket: S3 bucket for data
        instance_type: EC2 instance type for transform
        instance_count: Number of instances
        max_concurrent: Max concurrent transforms
        input_table: Athena table to read from
        lookback_hours: Hours of data to process
        row_limit: Maximum rows per batch
        
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
    safe_model_name = model_name.replace('/', '-').replace(':', '-')
    function_name = f'batch-transform-{safe_model_name}'[:64]
    rule_name = f'batch-transform-{safe_model_name}-rule'[:64]
    role_name = f'batch-transform-{safe_model_name}-role'[:64]
    
    # Default values
    if not s3_bucket:
        s3_bucket = f'fraud-detection-data-lake-skoppar-{account_id}'
    
    if not sagemaker_role:
        # Try to get from environment or use default
        import os
        sagemaker_role = os.getenv(
            'SAGEMAKER_EXEC_ROLE',
            f'arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-20250722T131288'
        )
    
    logger.info("=" * 60)
    logger.info(f"Creating scheduled batch transform for model: {model_name}")
    logger.info(f"  Schedule: {schedule_expression}")
    logger.info(f"  Region: {region}")
    logger.info(f"  Instance: {instance_type} x {instance_count}")
    logger.info("=" * 60)
    
    # Step 1: Create/get IAM role
    role_arn = get_or_create_lambda_role(iam_client, role_name, account_id)
    
    # Step 2: Create Lambda function
    logger.info(f"Creating Lambda function: {function_name}")
    
    deployment_package = create_lambda_deployment_package()
    
    environment = {
        'Variables': {
            'MODEL_NAME': model_name,
            'ATHENA_DATABASE': athena_database,
            'ATHENA_OUTPUT_S3': f's3://{s3_bucket}/athena-query-results/',
            'DATA_S3_BUCKET': s3_bucket,
            'SAGEMAKER_ROLE': sagemaker_role,
            'INSTANCE_TYPE': instance_type,
            'INSTANCE_COUNT': str(instance_count),
            'MAX_CONCURRENT': str(max_concurrent),
            'INPUT_TABLE': input_table,
            'LOOKBACK_HOURS': str(lookback_hours),
            'ROW_LIMIT': str(row_limit),
        }
    }
    
    try:
        # Try to update existing function
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=deployment_package
        )
        # Wait for update to complete
        time.sleep(5)
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Environment=environment,
            Timeout=900,  # 15 minutes
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
                Description=f'Scheduled batch transform for {model_name}',
                Timeout=900,  # 15 minutes
                MemorySize=512,
                Environment=environment,
                Tags={
                    'Purpose': 'ScheduledBatchTransform',
                    'Model': model_name
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
        Description=f'Trigger batch transform for {model_name}'
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
    logger.info("✓ Scheduled batch transform created successfully!")
    logger.info(f"  Lambda Function: {function_name}")
    logger.info(f"  EventBridge Rule: {rule_name}")
    logger.info(f"  Schedule: {schedule_expression}")
    logger.info(f"  Data Source: {athena_database}.{input_table}")
    logger.info(f"  Lookback: {lookback_hours} hours, max {row_limit} rows")
    logger.info("=" * 60)
    
    return {
        'function_name': function_name,
        'function_arn': function_arn,
        'rule_name': rule_name,
        'schedule': schedule_expression,
        'model_name': model_name,
        'instance_type': instance_type,
        'status': 'created'
    }


def delete_scheduled_batch_transform(model_name: str, region: str = 'us-east-1') -> dict:
    """Delete Lambda function and EventBridge rule."""
    lambda_client = boto3.client('lambda', region_name=region)
    events_client = boto3.client('events', region_name=region)
    iam_client = boto3.client('iam')
    
    safe_model_name = model_name.replace('/', '-').replace(':', '-')
    function_name = f'batch-transform-{safe_model_name}'[:64]
    rule_name = f'batch-transform-{safe_model_name}-rule'[:64]
    role_name = f'batch-transform-{safe_model_name}-role'[:64]
    
    logger.info(f"Deleting scheduled batch transform for: {model_name}")
    
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
        # Detach managed policies
        paginator = iam_client.get_paginator('list_attached_role_policies')
        for page in paginator.paginate(RoleName=role_name):
            for policy in page['AttachedPolicies']:
                iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
        
        # Delete inline policies
        paginator = iam_client.get_paginator('list_role_policies')
        for page in paginator.paginate(RoleName=role_name):
            for policy_name in page['PolicyNames']:
                iam_client.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
        
        iam_client.delete_role(RoleName=role_name)
        logger.info(f"  Deleted role: {role_name}")
    except ClientError:
        pass
    
    logger.info("✓ Scheduled batch transform deleted")
    return {'status': 'deleted', 'model_name': model_name}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup scheduled batch transform with EventBridge + Lambda",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create scheduled batch transform (daily at 2 AM UTC)
  python scripts/setup_scheduled_batch_transform.py \\
      --model-name sg-xgboost-fraud-detector \\
      --schedule "cron(0 2 * * ? *)"
  
  # Every 6 hours
  python scripts/setup_scheduled_batch_transform.py \\
      --model-name sg-xgboost-fraud-detector \\
      --schedule "rate(6 hours)"
  
  # With custom configuration
  python scripts/setup_scheduled_batch_transform.py \\
      --model-name sg-xgboost-fraud-detector \\
      --schedule "cron(0 2 * * ? *)" \\
      --instance-type ml.m5.2xlarge \\
      --instance-count 2 \\
      --lookback-hours 48 \\
      --row-limit 50000
  
  # Delete scheduled batch transform
  python scripts/setup_scheduled_batch_transform.py \\
      --model-name sg-xgboost-fraud-detector \\
      --delete

Schedule Expression Examples:
  rate(6 hours)        - Every 6 hours
  rate(1 day)          - Every day
  cron(0 2 * * ? *)    - Daily at 2 AM UTC
  cron(0 */6 * * ? *)  - Every 6 hours
  cron(0 2 ? * SUN *)  - Every Sunday at 2 AM UTC
        """
    )
    
    parser.add_argument(
        '--model-name',
        required=True,
        help='SageMaker model or model package group name'
    )
    parser.add_argument(
        '--schedule',
        default='cron(0 2 * * ? *)',
        help='EventBridge schedule expression (default: daily at 2 AM UTC)'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--sagemaker-role',
        help='SageMaker execution role ARN'
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
        '--instance-type',
        default='ml.m5.xlarge',
        help='EC2 instance type (default: ml.m5.xlarge)'
    )
    parser.add_argument(
        '--instance-count',
        type=int,
        default=1,
        help='Number of instances (default: 1)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=4,
        help='Max concurrent transforms (default: 4)'
    )
    parser.add_argument(
        '--input-table',
        default='training_data',
        help='Athena table to read from (default: training_data)'
    )
    parser.add_argument(
        '--lookback-hours',
        type=int,
        default=24,
        help='Hours of data to process (default: 24)'
    )
    parser.add_argument(
        '--row-limit',
        type=int,
        default=10000,
        help='Maximum rows per batch (default: 10000)'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete the scheduled batch transform instead of creating'
    )
    
    args = parser.parse_args()
    
    try:
        if args.delete:
            result = delete_scheduled_batch_transform(
                model_name=args.model_name,
                region=args.region
            )
        else:
            result = create_scheduled_batch_transform(
                model_name=args.model_name,
                schedule_expression=args.schedule,
                region=args.region,
                sagemaker_role=args.sagemaker_role,
                athena_database=args.athena_database,
                s3_bucket=args.s3_bucket,
                instance_type=args.instance_type,
                instance_count=args.instance_count,
                max_concurrent=args.max_concurrent,
                input_table=args.input_table,
                lookback_hours=args.lookback_hours,
                row_limit=args.row_limit
            )
        
        print(json.dumps(result, indent=2))
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
