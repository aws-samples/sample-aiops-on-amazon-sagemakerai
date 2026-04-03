"""
Lambda function for testing SageMaker endpoint inference.

This Lambda is invoked by the SageMaker Pipeline LambdaStep to:
- Test the deployed endpoint with sample data
- Measure latency and success rate
- Return test results
"""

import json
import logging
import time
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_runtime = boto3.client('sagemaker-runtime')


def lambda_handler(event, context):
    """Test inference endpoint and log results."""
    logger.info(f"Event: {json.dumps(event)}")
    
    endpoint_name = event.get('endpoint_name')
    num_samples = int(event.get('num_samples', 50))
    
    if not endpoint_name:
        raise ValueError("endpoint_name is required")
    
    results = {
        'total_invocations': 0,
        'successful_invocations': 0,
        'failed_invocations': 0,
        'latencies_ms': []
    }
    
    # Generate test samples (simplified - 33 features for fraud detection)
    test_samples = [
        {"features": [0.5] * 33}
        for _ in range(num_samples)
    ]
    
    for sample in test_samples:
        try:
            start = time.time()
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(sample)
            )
            latency = (time.time() - start) * 1000
            
            results['total_invocations'] += 1
            results['successful_invocations'] += 1
            results['latencies_ms'].append(latency)
            
        except Exception as e:
            logger.error(f"Invocation failed: {e}")
            results['total_invocations'] += 1
            results['failed_invocations'] += 1
    
    # Calculate statistics
    if results['latencies_ms']:
        latencies = results['latencies_ms']
        results['avg_latency_ms'] = sum(latencies) / len(latencies)
        results['min_latency_ms'] = min(latencies)
        results['max_latency_ms'] = max(latencies)
    
    logger.info(f"Test results: {json.dumps(results, default=str)}")
    
    return {
        'statusCode': 200,
        'endpoint_name': endpoint_name,
        'total_invocations': results['total_invocations'],
        'successful_invocations': results['successful_invocations'],
        'failed_invocations': results['failed_invocations'],
        'avg_latency_ms': results.get('avg_latency_ms', 0),
        'success_rate': results['successful_invocations'] / max(results['total_invocations'], 1)
    }
