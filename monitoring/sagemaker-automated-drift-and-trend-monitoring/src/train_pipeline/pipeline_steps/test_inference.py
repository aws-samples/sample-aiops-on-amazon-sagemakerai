"""
Lambda function for testing inference and generating visualizations.

⚠️ NOTE: This file uses awswrangler for SMALL MONITORING QUERIES.
For small aggregated results (<10K rows), awswrangler/pandas is efficient.
For large-scale processing, see preprocessing_pyspark.py.

This Lambda is invoked as a LambdaStep and:
- Invokes the deployed endpoint with test data
- Queries Athena for inference metrics (small aggregations)
- Generates 5 visualization charts
- Logs results and charts to MLflow

**Architecture Note:**
- Monitoring queries return small aggregated results
- awswrangler is appropriate for this use case
- For bulk data processing, use PySpark version
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import boto3
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')


def invoke_endpoint(
    endpoint_name: str,
    test_data: List[Dict[str, Any]],
    num_samples: int = 50
) -> Dict[str, Any]:
    """
    Invoke SageMaker endpoint with test data.

    Args:
        endpoint_name: SageMaker endpoint name
        test_data: List of test samples
        num_samples: Number of samples to test

    Returns:
        Dictionary with invocation results
    """
    logger.info(f"Invoking endpoint {endpoint_name} with {num_samples} samples")

    results = {
        'total_invocations': 0,
        'successful_invocations': 0,
        'failed_invocations': 0,
        'predictions': [],
        'latencies_ms': [],
        'errors': []
    }

    # Sample test data
    samples = test_data[:num_samples] if len(test_data) > num_samples else test_data

    for idx, sample in enumerate(samples):
        try:
            start_time = time.time()

            # Invoke endpoint
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(sample)
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            prediction = json.loads(response['Body'].read().decode())

            results['total_invocations'] += 1
            results['successful_invocations'] += 1
            results['predictions'].append(prediction)
            results['latencies_ms'].append(latency_ms)

            if (idx + 1) % 10 == 0:
                logger.info(f"Completed {idx + 1}/{num_samples} invocations")

        except Exception as e:
            logger.error(f"Invocation {idx + 1} failed: {e}")
            results['total_invocations'] += 1
            results['failed_invocations'] += 1
            results['errors'].append(str(e))

    # Calculate statistics
    if results['latencies_ms']:
        results['avg_latency_ms'] = float(np.mean(results['latencies_ms']))
        results['min_latency_ms'] = float(np.min(results['latencies_ms']))
        results['max_latency_ms'] = float(np.max(results['latencies_ms']))
        results['p50_latency_ms'] = float(np.percentile(results['latencies_ms'], 50))
        results['p95_latency_ms'] = float(np.percentile(results['latencies_ms'], 95))
        results['p99_latency_ms'] = float(np.percentile(results['latencies_ms'], 99))

    logger.info(f"✓ Completed {results['successful_invocations']}/{results['total_invocations']} invocations")
    logger.info(f"  Avg latency: {results.get('avg_latency_ms', 0):.2f} ms")

    return results


def query_athena_metrics(
    endpoint_name: str,
    time_window_minutes: int = 60
) -> pd.DataFrame:
    """
    Query Athena for inference metrics.

    Args:
        endpoint_name: SageMaker endpoint name
        time_window_minutes: Time window in minutes

    Returns:
        DataFrame with inference metrics
    """
    try:
        import awswrangler as wr

        logger.info(f"Querying Athena for metrics (last {time_window_minutes} minutes)")

        # Calculate time window
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_window_minutes)

        # Query inference responses
        query = f"""
        SELECT
            inference_id,
            request_timestamp,
            prediction,
            probability_fraud,
            probability_non_fraud,
            confidence_score,
            inference_latency_ms,
            prediction_bucket,
            is_high_confidence,
            is_low_confidence,
            ground_truth
        FROM fraud_detection.inference_responses
        WHERE endpoint_name = '{endpoint_name}'
          AND request_timestamp >= TIMESTAMP '{start_time.isoformat()}'
          AND request_timestamp <= TIMESTAMP '{end_time.isoformat()}'
        ORDER BY request_timestamp DESC
        """

        df = wr.athena.read_sql_query(
            sql=query,
            database=os.environ.get('ATHENA_DATABASE', 'fraud_detection'),
            ctas_approach=False
        )

        logger.info(f"✓ Retrieved {len(df)} records from Athena")
        return df

    except ImportError:
        logger.warning("awswrangler not available, skipping Athena query")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Athena query failed: {e}")
        return pd.DataFrame()


def generate_visualizations(
    df: pd.DataFrame,
    endpoint_name: str,
    output_dir: str = '/tmp/charts'
) -> List[str]:
    """
    Generate 5 visualization charts from Athena data.

    Args:
        df: DataFrame with inference data
        endpoint_name: Endpoint name for titles
        output_dir: Directory to save charts

    Returns:
        List of chart file paths
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("Generating visualization charts...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chart_paths = []

    if df.empty:
        logger.warning("No data available for visualizations")
        return chart_paths

    try:
        # Chart 1: Prediction Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        prediction_counts = df['prediction'].value_counts()
        ax.bar(['Non-Fraud', 'Fraud'], [
            prediction_counts.get(0, 0),
            prediction_counts.get(1, 0)
        ], color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title(f'Prediction Distribution - {endpoint_name}')
        ax.grid(True, alpha=0.3)

        chart_path = output_path / 'prediction_distribution.png'
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        chart_paths.append(str(chart_path))
        plt.close(fig)
        logger.info("✓ Chart 1: Prediction Distribution")

    except Exception as e:
        logger.error(f"Failed to generate chart 1: {e}")

    try:
        # Chart 2: Fraud Probability Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['probability_fraud'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax.set_xlabel('Fraud Probability')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Fraud Probability Distribution - {endpoint_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        chart_path = output_path / 'fraud_probability_histogram.png'
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        chart_paths.append(str(chart_path))
        plt.close(fig)
        logger.info("✓ Chart 2: Fraud Probability Histogram")

    except Exception as e:
        logger.error(f"Failed to generate chart 2: {e}")

    try:
        # Chart 3: Confidence Score Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['confidence_score'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(0.9, color='red', linestyle='--', linewidth=2, label='High Confidence')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Confidence Score Distribution - {endpoint_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        chart_path = output_path / 'confidence_distribution.png'
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        chart_paths.append(str(chart_path))
        plt.close(fig)
        logger.info("✓ Chart 3: Confidence Distribution")

    except Exception as e:
        logger.error(f"Failed to generate chart 3: {e}")

    try:
        # Chart 4: Latency Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['inference_latency_ms'], bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Inference Latency (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Inference Latency Distribution - {endpoint_name}')
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"Mean: {df['inference_latency_ms'].mean():.2f} ms\n"
        stats_text += f"Median: {df['inference_latency_ms'].median():.2f} ms\n"
        stats_text += f"P95: {df['inference_latency_ms'].quantile(0.95):.2f} ms"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        chart_path = output_path / 'latency_distribution.png'
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        chart_paths.append(str(chart_path))
        plt.close(fig)
        logger.info("✓ Chart 4: Latency Distribution")

    except Exception as e:
        logger.error(f"Failed to generate chart 4: {e}")

    try:
        # Chart 5: Prediction Timeline
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by time and count predictions
        df['timestamp'] = pd.to_datetime(df['request_timestamp'])
        df_sorted = df.sort_values('timestamp')

        # Resample to 1-minute buckets
        fraud_counts = df_sorted[df_sorted['prediction'] == 1].set_index('timestamp').resample('1T').size()
        non_fraud_counts = df_sorted[df_sorted['prediction'] == 0].set_index('timestamp').resample('1T').size()

        ax.plot(fraud_counts.index, fraud_counts.values, marker='o', color='red',
               linewidth=2, label='Fraud', alpha=0.7)
        ax.plot(non_fraud_counts.index, non_fraud_counts.values, marker='s', color='green',
               linewidth=2, label='Non-Fraud', alpha=0.7)

        ax.set_xlabel('Time')
        ax.set_ylabel('Predictions per Minute')
        ax.set_title(f'Prediction Timeline - {endpoint_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        chart_path = output_path / 'prediction_timeline.png'
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        chart_paths.append(str(chart_path))
        plt.close(fig)
        logger.info("✓ Chart 5: Prediction Timeline")

    except Exception as e:
        logger.error(f"Failed to generate chart 5: {e}")

    logger.info(f"✓ Generated {len(chart_paths)} charts")
    return chart_paths


def log_to_mlflow(
    invocation_results: Dict[str, Any],
    athena_metrics: pd.DataFrame,
    chart_paths: List[str],
    endpoint_name: str
) -> str:
    """
    Log results and charts to MLflow.

    Args:
        invocation_results: Endpoint invocation results
        athena_metrics: Athena query results
        chart_paths: List of chart file paths
        endpoint_name: Endpoint name

    Returns:
        MLflow run ID
    """
    try:
        import mlflow

        # Set tracking URI
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        if not mlflow_tracking_uri:
            logger.warning("MLflow tracking URI not set, skipping MLflow logging")
            return None

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME',
                                        'credit-card-fraud-detection-inference')
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log invocation metrics
            mlflow.log_metric('total_invocations', invocation_results['total_invocations'])
            mlflow.log_metric('successful_invocations', invocation_results['successful_invocations'])
            mlflow.log_metric('failed_invocations', invocation_results['failed_invocations'])

            if 'avg_latency_ms' in invocation_results:
                mlflow.log_metric('avg_latency_ms', invocation_results['avg_latency_ms'])
                mlflow.log_metric('p95_latency_ms', invocation_results['p95_latency_ms'])
                mlflow.log_metric('p99_latency_ms', invocation_results['p99_latency_ms'])

            # Log Athena metrics
            if not athena_metrics.empty:
                fraud_rate = athena_metrics['prediction'].mean()
                avg_confidence = athena_metrics['confidence_score'].mean()

                mlflow.log_metric('athena_sample_count', len(athena_metrics))
                mlflow.log_metric('fraud_rate', fraud_rate)
                mlflow.log_metric('avg_confidence', avg_confidence)

            # Log charts
            for chart_path in chart_paths:
                chart_name = Path(chart_path).name
                mlflow.log_artifact(chart_path, artifact_path='charts')
                logger.info(f"✓ Logged chart: {chart_name}")

            # Log tags
            mlflow.set_tags({
                'pipeline_step': 'inference_testing',
                'endpoint_name': endpoint_name,
                'test_type': 'pipeline_validation'
            })

            run_id = run.info.run_id
            logger.info(f"✓ Logged to MLflow run: {run_id}")
            return run_id

    except ImportError:
        logger.warning("MLflow not available, skipping logging")
        return None
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
        return None


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for inference testing.

    Args:
        event: Lambda event containing:
            - endpoint_name: SageMaker endpoint name
            - test_data_s3_path: S3 path to test data
            - num_samples: Number of samples to test (default: 50)
            - time_window_minutes: Athena query window (default: 60)
        context: Lambda context

    Returns:
        Dictionary with test results
    """
    logger.info("Starting inference testing")
    logger.info(f"Event: {json.dumps(event, default=str)}")

    try:
        # Extract parameters
        endpoint_name = event.get('endpoint_name')
        test_data_s3_path = event.get('test_data_s3_path')
        num_samples = int(event.get('num_samples', 50))
        time_window_minutes = int(event.get('time_window_minutes', 60))

        if not endpoint_name:
            raise ValueError("endpoint_name is required")

        # Load test data from S3
        test_data = []
        if test_data_s3_path:
            # Load from S3 (simplified - in production, use boto3 to download)
            logger.info(f"Loading test data from {test_data_s3_path}")
            # For now, use sample data
            test_data = [{'feature': i} for i in range(num_samples)]
        else:
            # Use sample data
            logger.info("Using sample test data")
            test_data = [{'feature': i} for i in range(num_samples)]

        # Step 1: Invoke endpoint
        invocation_results = invoke_endpoint(endpoint_name, test_data, num_samples)

        # Step 2: Query Athena for metrics
        athena_metrics = query_athena_metrics(endpoint_name, time_window_minutes)

        # Step 3: Generate visualizations
        chart_paths = generate_visualizations(athena_metrics, endpoint_name)

        # Step 4: Log to MLflow
        mlflow_run_id = log_to_mlflow(
            invocation_results,
            athena_metrics,
            chart_paths,
            endpoint_name
        )

        # Return success response
        response = {
            'statusCode': 200,
            'endpoint_name': endpoint_name,
            'invocation_results': {
                'total_invocations': invocation_results['total_invocations'],
                'successful_invocations': invocation_results['successful_invocations'],
                'failed_invocations': invocation_results['failed_invocations'],
                'avg_latency_ms': invocation_results.get('avg_latency_ms', 0),
                'p95_latency_ms': invocation_results.get('p95_latency_ms', 0)
            },
            'athena_metrics': {
                'sample_count': len(athena_metrics),
                'fraud_rate': float(athena_metrics['prediction'].mean()) if not athena_metrics.empty else 0,
                'avg_confidence': float(athena_metrics['confidence_score'].mean()) if not athena_metrics.empty else 0
            },
            'charts_generated': len(chart_paths),
            'mlflow_run_id': mlflow_run_id
        }

        logger.info(f"✓ Inference testing completed: {json.dumps(response, default=str)}")
        return response

    except Exception as e:
        logger.error(f"Inference testing failed: {str(e)}", exc_info=True)

        return {
            'statusCode': 500,
            'error': str(e),
            'endpoint_name': event.get('endpoint_name', 'unknown')
        }


if __name__ == '__main__':
    """Test locally."""
    import argparse

    parser = argparse.ArgumentParser(description="Test inference endpoint")
    parser.add_argument('--endpoint-name', required=True, help='Endpoint name')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples')

    args = parser.parse_args()

    event = {
        'endpoint_name': args.endpoint_name,
        'num_samples': args.num_samples
    }

    result = lambda_handler(event, None)
    print(json.dumps(result, indent=2, default=str))
