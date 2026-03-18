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


def lambda_handler(event, context):
    """Deploy SageMaker endpoint."""
    logger.info(f"Event: {json.dumps(event)}")
    
    model_name = event.get('model_name')
    endpoint_name = event.get('endpoint_name')
    memory_size_mb = int(event.get('memory_size_mb', 4096))
    max_concurrency = int(event.get('max_concurrency', 20))
    
    if not model_name or not endpoint_name:
        raise ValueError("model_name and endpoint_name are required")
    
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
        
        return {
            'statusCode': 200,
            'endpoint_name': endpoint_name,
            'endpoint_arn': response['EndpointArn'],
            'status': status,
            'action': action
        }
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return {'statusCode': 500, 'error': str(e)}
