import boto3
import io
import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / '.env')

from src.config.config import (
    SQS_QUEUE_NAME, SAGEMAKER_EXEC_ROLE, AWS_DEFAULT_REGION,
    ATHENA_DATABASE, ATHENA_OUTPUT_S3, LAMBDA_LOGGER_NAME
)


def setup_sqs_queue():
    sqs = boto3.client('sqs', region_name=AWS_DEFAULT_REGION)
    try:
        response = sqs.create_queue(
            QueueName=SQS_QUEUE_NAME,
            Attributes={'VisibilityTimeout': '300', 'MessageRetentionPeriod': '86400'}
        )
        return response['QueueUrl']
    except sqs.exceptions.QueueNameExists:
        return sqs.get_queue_url(QueueName=SQS_QUEUE_NAME)['QueueUrl']


def setup_lambda_consumer(role_arn, queue_arn):
    """
    Create Lambda function to consume SQS messages and log to Athena.

    Returns:
        Lambda function ARN
    """
    lam = boto3.client('lambda', region_name=AWS_DEFAULT_REGION)
    func_name = LAMBDA_LOGGER_NAME

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        # Lambda logger is in src/drift_monitoring/
        lambda_file = Path(__file__).parent.parent / 'drift_monitoring' / 'lambda_inference_logger.py'
        if not lambda_file.exists():
            raise FileNotFoundError(f"Lambda logger not found at: {lambda_file}")
        zf.write(lambda_file, 'lambda_inference_logger.py')
    buf.seek(0)
    code = buf.read()

    lambda_arn = None
    try:
        response = lam.create_function(
            FunctionName=func_name,
            Runtime='python3.12',
            Role=role_arn,
            Handler='lambda_inference_logger.lambda_handler',
            Code={'ZipFile': code},
            Timeout=60,
            Environment={'Variables': {
                'ATHENA_DATABASE': ATHENA_DATABASE,
                'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
            }}
        )
        lambda_arn = response['FunctionArn']
    except lam.exceptions.ResourceConflictException:
        # Function already exists, update it
        buf.seek(0)
        lam.update_function_code(FunctionName=func_name, ZipFile=code)
        # Get function ARN
        response = lam.get_function(FunctionName=func_name)
        lambda_arn = response['Configuration']['FunctionArn']

    try:
        lam.create_event_source_mapping(
            EventSourceArn=queue_arn,
            FunctionName=func_name,
            BatchSize=10,
            MaximumBatchingWindowInSeconds=30,
        )
    except lam.exceptions.ResourceConflictException:
        pass

    return lambda_arn


if __name__ == '__main__':
    account_id = boto3.client('sts').get_caller_identity()['Account']

    queue_url = setup_sqs_queue()
    queue_arn = f"arn:aws:sqs:{AWS_DEFAULT_REGION}:{account_id}:{SQS_QUEUE_NAME}"
    setup_lambda_consumer(SAGEMAKER_EXEC_ROLE, queue_arn)

    print(f"SQS Queue URL: {queue_url}")
    print(f"Lambda consumer created with SQS trigger")