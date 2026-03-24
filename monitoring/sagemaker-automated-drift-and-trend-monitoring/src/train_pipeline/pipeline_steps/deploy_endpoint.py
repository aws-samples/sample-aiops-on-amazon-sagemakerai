"""
Lambda function for deploying SageMaker endpoint in pipeline.

This Lambda is invoked as a LambdaStep and:
- Creates or updates SageMaker endpoint
- Configures with Athena logging environment variables
- Uses inference_handler.py for inference
- Supports serverless inference configuration
"""

import json
import logging
import os
import time
from typing import Dict, Any

import boto3
from botocore.exceptions import ClientError

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for endpoint deployment.

    Args:
        event: Lambda event containing:
            - model_name: SageMaker model name
            - endpoint_name: SageMaker endpoint name
            - endpoint_config_name: Endpoint configuration name
            - memory_size_mb: Serverless memory (default: 4096)
            - max_concurrency: Serverless max concurrency (default: 20)
            - enable_athena_logging: Enable Athena logging (default: true)
            - mlflow_run_id: MLflow run ID for tracking
            - model_version: Model version tag
        context: Lambda context

    Returns:
        Dictionary with endpoint details
    """
    logger.info("Starting endpoint deployment")
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Extract parameters
        model_name = event.get('model_name')
        endpoint_name = event.get('endpoint_name')
        endpoint_config_name = event.get('endpoint_config_name')
        memory_size_mb = int(event.get('memory_size_mb', 4096))
        max_concurrency = int(event.get('max_concurrency', 20))
        enable_athena_logging = event.get('enable_athena_logging', True)
        mlflow_run_id = event.get('mlflow_run_id', 'unknown')
        model_version = event.get('model_version', 'v1.0')

        # Validate required parameters
        if not model_name:
            raise ValueError("model_name is required")
        if not endpoint_name:
            raise ValueError("endpoint_name is required")

        # Generate endpoint config name if not provided
        if not endpoint_config_name:
            endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"

        # Step 1: Create endpoint configuration
        logger.info(f"Creating endpoint configuration: {endpoint_config_name}")

        # Environment variables for inference handler
        environment = {
            'ENABLE_ATHENA_LOGGING': 'true' if enable_athena_logging else 'false',
            'ENDPOINT_NAME': endpoint_name,
            'MODEL_VERSION': model_version,
            'MLFLOW_RUN_ID': mlflow_run_id,
            'ATHENA_DATABASE': os.environ.get('ATHENA_DATABASE', 'fraud_detection'),
            'ATHENA_OUTPUT_S3': os.environ.get('ATHENA_OUTPUT_S3',
                's3://fraud-detection-data-lake/athena-query-results/'),
            'INFERENCE_LOG_BATCH_SIZE': os.environ.get('INFERENCE_LOG_BATCH_SIZE', '50'),  # Flush after 50 predictions
            'INFERENCE_LOG_FLUSH_INTERVAL': os.environ.get('INFERENCE_LOG_FLUSH_INTERVAL', '300'),
        }

        # Create serverless config
        production_variants = [{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'ServerlessConfig': {
                'MemorySizeInMB': memory_size_mb,
                'MaxConcurrency': max_concurrency
            }
        }]

        # Create endpoint configuration
        config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=production_variants,
            Tags=[
                {'Key': 'MLflowRunId', 'Value': mlflow_run_id},
                {'Key': 'ModelVersion', 'Value': model_version},
                {'Key': 'Pipeline', 'Value': 'fraud-detection'},
                {'Key': 'AthenaLogging', 'Value': str(enable_athena_logging)}
            ]
        )

        logger.info(f"✓ Endpoint configuration created: {endpoint_config_name}")

        # Step 2: Create or update endpoint
        try:
            # Check if endpoint exists
            describe_response = sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )

            # Endpoint exists, update it
            logger.info(f"Endpoint {endpoint_name} exists, updating...")

            update_response = sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                RetainAllVariantProperties=False
            )

            logger.info(f"✓ Endpoint update initiated: {endpoint_name}")
            action = 'updated'

        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                # Endpoint doesn't exist, create it
                logger.info(f"Creating new endpoint: {endpoint_name}")

                create_response = sagemaker_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name,
                    Tags=[
                        {'Key': 'MLflowRunId', 'Value': mlflow_run_id},
                        {'Key': 'ModelVersion', 'Value': model_version},
                        {'Key': 'Pipeline', 'Value': 'fraud-detection'}
                    ]
                )

                logger.info(f"✓ Endpoint creation initiated: {endpoint_name}")
                action = 'created'
            else:
                raise

        # Step 3: Wait for endpoint to be in service (with timeout)
        logger.info(f"Waiting for endpoint to be in service...")

        max_wait_time = 600  # 10 minutes
        start_time = time.time()

        while (time.time() - start_time) < max_wait_time:
            status_response = sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )

            status = status_response['EndpointStatus']
            logger.info(f"Endpoint status: {status}")

            if status == 'InService':
                logger.info(f"✓ Endpoint is in service: {endpoint_name}")
                break
            elif status in ['Failed', 'RolledBack']:
                error_msg = f"Endpoint deployment failed with status: {status}"
                if 'FailureReason' in status_response:
                    error_msg += f" - {status_response['FailureReason']}"
                logger.error(error_msg)
                raise Exception(error_msg)

            # Wait before checking again
            time.sleep(30)
        else:
            # Timeout reached
            raise Exception(f"Endpoint deployment timed out after {max_wait_time} seconds")

        # Return success response
        response = {
            'statusCode': 200,
            'action': action,
            'endpoint_name': endpoint_name,
            'endpoint_config_name': endpoint_config_name,
            'endpoint_arn': status_response['EndpointArn'],
            'status': status_response['EndpointStatus'],
            'model_name': model_name,
            'memory_size_mb': memory_size_mb,
            'max_concurrency': max_concurrency,
            'athena_logging_enabled': enable_athena_logging,
            'mlflow_run_id': mlflow_run_id,
            'model_version': model_version
        }

        logger.info(f"Deployment completed successfully: {json.dumps(response)}")
        return response

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}", exc_info=True)

        return {
            'statusCode': 500,
            'error': str(e),
            'endpoint_name': event.get('endpoint_name', 'unknown')
        }


def create_or_update_endpoint_sync(
    model_name: str,
    endpoint_name: str,
    memory_size_mb: int = 4096,
    max_concurrency: int = 20,
    enable_athena_logging: bool = True,
    mlflow_run_id: str = 'unknown',
    model_version: str = 'v1.0'
) -> Dict[str, Any]:
    """
    Synchronous wrapper for testing outside Lambda.

    Args:
        model_name: SageMaker model name
        endpoint_name: SageMaker endpoint name
        memory_size_mb: Serverless memory size
        max_concurrency: Serverless max concurrency
        enable_athena_logging: Enable Athena logging
        mlflow_run_id: MLflow run ID
        model_version: Model version tag

    Returns:
        Deployment response dictionary
    """
    event = {
        'model_name': model_name,
        'endpoint_name': endpoint_name,
        'memory_size_mb': memory_size_mb,
        'max_concurrency': max_concurrency,
        'enable_athena_logging': enable_athena_logging,
        'mlflow_run_id': mlflow_run_id,
        'model_version': model_version
    }

    return lambda_handler(event, None)


if __name__ == '__main__':
    """Test endpoint deployment locally."""
    import sys

    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python deploy_endpoint.py <model_name> <endpoint_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    endpoint_name = sys.argv[2]

    result = create_or_update_endpoint_sync(
        model_name=model_name,
        endpoint_name=endpoint_name
    )

    print(json.dumps(result, indent=2))
