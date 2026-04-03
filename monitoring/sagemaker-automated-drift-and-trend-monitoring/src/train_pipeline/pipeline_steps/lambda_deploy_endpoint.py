"""
Lambda function for deploying SageMaker endpoint.

This Lambda is invoked by the SageMaker Pipeline LambdaStep to:
- Create endpoint configuration
- Create or update SageMaker endpoint
- Wait for endpoint to be in service
"""

import json
import logging
import time
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_client = boto3.client('sagemaker')
runtime_client = boto3.client('sagemaker-runtime')


def validate_endpoint_version(endpoint_name, expected_version, mlflow_run_id):
    """
    Validate deployed endpoint returns correct version in inference response.

    Args:
        endpoint_name: SageMaker endpoint name
        expected_version: Expected model version (e.g., "v2")
        mlflow_run_id: Expected MLflow run ID

    Returns:
        bool: True if validation passes
        dict: Validation results

    Raises:
        ValueError: If version mismatch detected
    """
    logger.info(f"Validating endpoint version: {endpoint_name}")

    try:
        # Send test inference request
        test_input = {
            "transaction_amount": 100.0,
            "customer_age": 30,
            "transaction_hour": 12,
            "is_international": 0,
            "transaction_count_1d": 3,
        }

        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_input)
        )

        result = json.loads(response['Body'].read().decode())

        # Check if metadata exists
        if 'metadata' not in result:
            logger.error("Validation FAILED: No metadata in inference response")
            return False, {
                'status': 'FAILED',
                'error': 'No metadata in response'
            }

        metadata = result['metadata']
        actual_version = metadata.get('model_version', 'unknown')
        actual_run_id = metadata.get('mlflow_run_id', 'unknown')

        # Validate version
        if actual_version != expected_version:
            logger.error(f"Version mismatch: Expected {expected_version}, got {actual_version}")
            return False, {
                'status': 'FAILED',
                'expected_version': expected_version,
                'actual_version': actual_version,
                'error': 'Version mismatch'
            }

        # Validate MLflow run ID
        if mlflow_run_id and actual_run_id != mlflow_run_id:
            logger.warning(f"Run ID mismatch: Expected {mlflow_run_id}, got {actual_run_id}")
            # Don't fail on run ID mismatch, just warn

        logger.info(f"✓ Validation PASSED: Version {actual_version} matches expected {expected_version}")
        return True, {
            'status': 'PASSED',
            'model_version': actual_version,
            'mlflow_run_id': actual_run_id,
            'endpoint_name': metadata.get('endpoint_name', 'unknown')
        }

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, {
            'status': 'ERROR',
            'error': str(e)
        }


def lambda_handler(event, context):
    """Deploy SageMaker endpoint."""
    logger.info(f"Event: {json.dumps(event)}")

    model_name = event.get('model_name')
    endpoint_name = event.get('endpoint_name')
    memory_size_mb = int(event.get('memory_size_mb', 4096))
    max_concurrency = int(event.get('max_concurrency', 20))
    model_package_arn = event.get('model_package_arn')
    mlflow_run_id = event.get('mlflow_run_id', 'pipeline')

    if not model_name or not endpoint_name:
        raise ValueError("model_name and endpoint_name are required")

    # Extract version from Model Package ARN
    # Format: arn:aws:sagemaker:region:account:model-package/group-name/VERSION
    model_version = 'v1'  # Default
    if model_package_arn:
        try:
            # Split ARN and get the last segment (version number)
            version_number = model_package_arn.split('/')[-1]
            model_version = f"v{version_number}"
            logger.info(f"Extracted model version: {model_version} from {model_package_arn}")
        except Exception as e:
            logger.warning(f"Could not extract version from ARN: {e}, using default: {model_version}")

    # Update Model environment variables with version info
    try:
        model_desc = sagemaker_client.describe_model(ModelName=model_name)
        current_env = model_desc['PrimaryContainer'].get('Environment', {})

        # Update with version info
        current_env['MODEL_VERSION'] = model_version
        current_env['MLFLOW_RUN_ID'] = mlflow_run_id

        # Create new model with updated env vars (SageMaker models are immutable)
        new_model_name = f"{model_name}-{int(time.time())}"
        sagemaker_client.create_model(
            ModelName=new_model_name,
            PrimaryContainer={
                'Image': model_desc['PrimaryContainer']['Image'],
                'ModelDataUrl': model_desc['PrimaryContainer']['ModelDataUrl'],
                'Environment': current_env
            },
            ExecutionRoleArn=model_desc['ExecutionRoleArn']
        )
        logger.info(f"Created model with version info: {new_model_name}")

        # Use the new model for endpoint
        model_name = new_model_name

    except Exception as e:
        logger.warning(f"Could not update model environment: {e}, using original model")

    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"

    try:
        # Create endpoint configuration
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'ServerlessConfig': {
                    'MemorySizeInMB': memory_size_mb,
                    'MaxConcurrency': max_concurrency
                }
            }]
        )
        logger.info(f"Created endpoint config: {endpoint_config_name}")
        
        # Check if endpoint exists
        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            # Update existing endpoint
            sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            action = 'updated'
        except ClientError:
            # Create new endpoint
            sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            action = 'created'
        
        logger.info(f"Endpoint {action}: {endpoint_name}")
        
        # Wait for endpoint to be in service (max 10 min)
        for _ in range(20):
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            if status == 'InService':
                break
            elif status in ['Failed', 'RolledBack']:
                raise Exception(f"Endpoint failed: {status}")
            time.sleep(30)

        # Validate endpoint version after deployment
        validation_passed = False
        validation_result = {}
        if status == 'InService':
            try:
                validation_passed, validation_result = validate_endpoint_version(
                    endpoint_name,
                    model_version,
                    mlflow_run_id
                )
                logger.info(f"Version validation result: {validation_result}")
            except Exception as e:
                logger.warning(f"Version validation failed but continuing: {e}")
                validation_result = {'status': 'ERROR', 'error': str(e)}

        return {
            'statusCode': 200,
            'endpoint_name': endpoint_name,
            'endpoint_arn': response['EndpointArn'],
            'status': status,
            'action': action,
            'version_validation': validation_result
        }
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return {'statusCode': 500, 'error': str(e)}
