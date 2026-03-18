"""
Enhanced endpoint testing with Athena analytics.

Tests SageMaker endpoints and queries Athena for comprehensive metrics.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = _project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

import boto3
import pandas as pd
import numpy as np

from src.config.config import (
    CSV_TRAINING_DATA,
    ATHENA_TRAINING_TABLE,
    ATHENA_DATABASE,
)
from src.pipeline.athena.athena_client import AthenaClient

logger = logging.getLogger(__name__)


def load_test_data(
    data_source: str = "csv",
    test_data_path: Optional[str] = None,
    num_samples: int = 100,
) -> pd.DataFrame:
    """
    Load test data for endpoint testing.

    Args:
        data_source: 'csv' or 'athena'
        test_data_path: Path to CSV (if data_source='csv')
        num_samples: Number of samples to load

    Returns:
        DataFrame with test data
    """
    if data_source == "athena":
        logger.info(f"Loading {num_samples} samples from Athena")
        client = AthenaClient()
        df = client.read_table(
            ATHENA_TRAINING_TABLE,
            limit=num_samples
        )
    else:
        csv_path = Path(test_data_path) if test_data_path else CSV_TRAINING_DATA
        logger.info(f"Loading {num_samples} samples from {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.sample(n=min(num_samples, len(df)), random_state=42)

    return df


def invoke_endpoint(
    endpoint_name: str,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Invoke SageMaker endpoint with input data.

    Args:
        endpoint_name: SageMaker endpoint name
        input_data: Input features as dictionary

    Returns:
        Prediction results
    """
    runtime_client = boto3.client('sagemaker-runtime')

    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(input_data)
    )

    result = json.loads(response['Body'].read().decode())
    return result


def test_endpoint_realtime(
    endpoint_name: str,
    test_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Test endpoint with real-time predictions.

    Args:
        endpoint_name: SageMaker endpoint name
        test_data: DataFrame with test samples

    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing endpoint with {len(test_data)} samples")

    predictions = []
    latencies = []
    errors = []

    for idx, row in test_data.iterrows():
        try:
            # Prepare input (drop target column if present)
            input_dict = row.drop(['is_fraud', 'fraud_prediction', 'fraud_probability'],
                                 errors='ignore').to_dict()

            # Invoke endpoint
            start_time = time.time()
            result = invoke_endpoint(endpoint_name, input_dict)
            latency = (time.time() - start_time) * 1000

            predictions.append(result)
            latencies.append(latency)

        except Exception as e:
            logger.error(f"Error predicting sample {idx}: {e}")
            errors.append(str(e))

    # Calculate statistics
    results = {
        'total_predictions': len(predictions),
        'successful_predictions': len(predictions),
        'failed_predictions': len(errors),
        'avg_latency_ms': float(np.mean(latencies)) if latencies else 0,
        'min_latency_ms': float(np.min(latencies)) if latencies else 0,
        'max_latency_ms': float(np.max(latencies)) if latencies else 0,
        'p50_latency_ms': float(np.percentile(latencies, 50)) if latencies else 0,
        'p95_latency_ms': float(np.percentile(latencies, 95)) if latencies else 0,
        'p99_latency_ms': float(np.percentile(latencies, 99)) if latencies else 0,
        'predictions': predictions,
        'errors': errors,
    }

    return results


def query_athena_metrics(
    endpoint_name: str,
    time_window_minutes: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Query Athena for aggregated inference metrics.

    Args:
        endpoint_name: SageMaker endpoint name
        time_window_minutes: Time window for metrics

    Returns:
        Dictionary with aggregated metrics or None if no data
    """
    try:
        client = AthenaClient()

        # Query for aggregated metrics
        query = f"""
        SELECT
            COUNT(*) as total_predictions,
            AVG(probability_fraud) as avg_fraud_prob,
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as fraud_count,
            SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as non_fraud_count,
            AVG(inference_latency_ms) as avg_latency_ms,
            STDDEV(inference_latency_ms) as std_latency_ms,
            MAX(inference_latency_ms) as max_latency_ms,
            MIN(inference_latency_ms) as min_latency_ms,
            SUM(CASE WHEN is_high_confidence THEN 1 ELSE 0 END) as high_confidence_count,
            SUM(CASE WHEN is_low_confidence THEN 1 ELSE 0 END) as low_confidence_count,
            COUNT(DISTINCT DATE_TRUNC('minute', request_timestamp)) as active_minutes
        FROM {ATHENA_DATABASE}.inference_responses
        WHERE endpoint_name = '{endpoint_name}'
          AND request_timestamp > CURRENT_TIMESTAMP - INTERVAL '{time_window_minutes}' MINUTE
        """

        logger.info(f"Querying Athena for metrics (last {time_window_minutes} minutes)")
        df = client.execute_query(query)

        if df.empty or df['total_predictions'].iloc[0] == 0:
            logger.warning("No inference data found in Athena")
            return None

        # Convert to dictionary
        metrics = df.iloc[0].to_dict()

        # Calculate derived metrics
        if metrics['total_predictions'] > 0:
            metrics['fraud_rate'] = metrics['fraud_count'] / metrics['total_predictions']
            metrics['high_confidence_rate'] = metrics['high_confidence_count'] / metrics['total_predictions']
            metrics['low_confidence_rate'] = metrics['low_confidence_count'] / metrics['total_predictions']

        return metrics

    except Exception as e:
        logger.error(f"Error querying Athena: {e}")
        return None


def print_test_results(
    realtime_results: Dict[str, Any],
    athena_metrics: Optional[Dict[str, Any]] = None,
):
    """Print formatted test results."""
    print("\n" + "=" * 80)
    print("ENDPOINT TEST RESULTS")
    print("=" * 80)

    print("\nReal-time Testing:")
    print(f"  Total Predictions: {realtime_results['total_predictions']}")
    print(f"  Successful: {realtime_results['successful_predictions']}")
    print(f"  Failed: {realtime_results['failed_predictions']}")

    print("\nLatency Statistics:")
    print(f"  Average: {realtime_results['avg_latency_ms']:.2f} ms")
    print(f"  Min: {realtime_results['min_latency_ms']:.2f} ms")
    print(f"  Max: {realtime_results['max_latency_ms']:.2f} ms")
    print(f"  P50: {realtime_results['p50_latency_ms']:.2f} ms")
    print(f"  P95: {realtime_results['p95_latency_ms']:.2f} ms")
    print(f"  P99: {realtime_results['p99_latency_ms']:.2f} ms")

    if realtime_results['predictions']:
        print("\nSample Predictions:")
        for i, pred in enumerate(realtime_results['predictions'][:5], 1):
            fraud_prob = pred['probabilities']['fraud'][0]
            prediction = pred['predictions'][0]
            print(f"  Sample {i}: Fraud={prediction}, Probability={fraud_prob:.4f}")

    if athena_metrics:
        print("\n" + "=" * 80)
        print("ATHENA ANALYTICS (Last 60 minutes)")
        print("=" * 80)

        print(f"\nTotal Predictions Logged: {athena_metrics['total_predictions']}")
        print(f"Fraud Predictions: {athena_metrics['fraud_count']}")
        print(f"Non-Fraud Predictions: {athena_metrics['non_fraud_count']}")
        print(f"Fraud Rate: {athena_metrics.get('fraud_rate', 0):.2%}")

        print(f"\nConfidence Distribution:")
        print(f"  High Confidence: {athena_metrics['high_confidence_count']} ({athena_metrics.get('high_confidence_rate', 0):.2%})")
        print(f"  Low Confidence: {athena_metrics['low_confidence_count']} ({athena_metrics.get('low_confidence_rate', 0):.2%})")

        print(f"\nAverage Fraud Probability: {athena_metrics['avg_fraud_prob']:.4f}")
        print(f"Average Latency: {athena_metrics['avg_latency_ms']:.2f} ms")

    print("\n" + "=" * 80)


def test_endpoint(
    endpoint_name: str,
    num_samples: int = 100,
    data_source: str = "csv",
    test_data_path: Optional[str] = None,
    enable_analytics: bool = True,
    generate_charts: bool = False,
    log_to_mlflow: bool = False,
    time_window_minutes: int = 60,
) -> Dict[str, Any]:
    """
    Test SageMaker endpoint with comprehensive analytics.

    Args:
        endpoint_name: SageMaker endpoint name
        num_samples: Number of test samples
        data_source: 'csv' or 'athena'
        test_data_path: Path to CSV (if using CSV)
        enable_analytics: Query Athena for analytics
        generate_charts: Generate visualization charts
        log_to_mlflow: Log charts and metrics to MLflow
        time_window_minutes: Time window for Athena metrics

    Returns:
        Dictionary with complete test results
    """
    print("=" * 80)
    print("SAGEMAKER ENDPOINT TESTING")
    print("=" * 80)
    print(f"Endpoint: {endpoint_name}")
    print(f"Samples: {num_samples}")
    print(f"Data Source: {data_source}")
    print("=" * 80 + "\n")

    # Load test data
    test_data = load_test_data(
        data_source=data_source,
        test_data_path=test_data_path,
        num_samples=num_samples,
    )

    # Test endpoint
    realtime_results = test_endpoint_realtime(endpoint_name, test_data)

    # Wait a bit for logs to flush to Athena
    if enable_analytics:
        print("\nWaiting 45 seconds for inference logs to flush to Athena...")
        time.sleep(45)

        # Query Athena metrics
        athena_metrics = query_athena_metrics(endpoint_name, time_window_minutes)
    else:
        athena_metrics = None

    # Print results
    print_test_results(realtime_results, athena_metrics)

    # Generate charts and log to MLflow
    chart_metrics = None
    if generate_charts and enable_analytics:
        try:
            from src.utils.visualization_utils import log_all_charts_to_mlflow

            print("\nGenerating visualization charts...")
            client = AthenaClient()

            if log_to_mlflow:
                # Log to new MLflow run
                import mlflow
                from src.config.config import MLFLOW_TRACKING_URI
                from src.utils.mlflow_utils import setup_mlflow_tracking, get_or_create_experiment
                from src.config.config import MLFLOW_INFERENCE_EXPERIMENT_NAME

                setup_mlflow_tracking(MLFLOW_TRACKING_URI)
                experiment_id = get_or_create_experiment(MLFLOW_INFERENCE_EXPERIMENT_NAME)

                with mlflow.start_run(experiment_id=experiment_id, run_name=f"test-{endpoint_name}"):
                    chart_metrics = log_all_charts_to_mlflow(
                        client,
                        endpoint_name,
                        mlflow_run_id=None,
                        days=7,
                    )
                    mlflow.log_param("endpoint_name", endpoint_name)
                    mlflow.log_param("test_samples", num_samples)

                print("✓ Charts generated and logged to MLflow")
            else:
                # Just generate charts without MLflow logging
                from src.utils.visualization_utils import (
                    create_roc_curve_from_athena,
                    create_confusion_matrix_from_athena,
                    create_prediction_distribution,
                    create_latency_heatmap,
                    create_confidence_distribution,
                )

                print("\nGenerating charts (not logging to MLflow)...")
                chart_metrics = {}

                try:
                    fig, metrics = create_prediction_distribution(client, endpoint_name)
                    chart_metrics['distribution'] = metrics
                    print(f"  ✓ Prediction distribution: {metrics.get('total_predictions', 0)} predictions")
                except Exception as e:
                    print(f"  ✗ Prediction distribution failed: {e}")

                try:
                    fig, metrics = create_latency_heatmap(client, endpoint_name)
                    chart_metrics['latency'] = metrics
                    print(f"  ✓ Latency heatmap: {metrics.get('avg_latency_ms', 0):.2f}ms avg")
                except Exception as e:
                    print(f"  ✗ Latency heatmap failed: {e}")

                try:
                    fig, metrics = create_confidence_distribution(client, endpoint_name)
                    chart_metrics['confidence'] = metrics
                    print(f"  ✓ Confidence distribution: {metrics.get('avg_confidence', 0):.3f} avg")
                except Exception as e:
                    print(f"  ✗ Confidence distribution failed: {e}")

        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            chart_metrics = None

    # Return combined results
    return {
        'realtime': realtime_results,
        'athena': athena_metrics,
        'charts': chart_metrics,
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Test SageMaker endpoint")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint name")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--data-source", choices=["csv", "athena"], default="csv")
    parser.add_argument("--test-data-path", help="Path to CSV test data")
    parser.add_argument("--disable-analytics", action="store_true",
                       help="Disable Athena analytics")
    parser.add_argument("--generate-charts", action="store_true",
                       help="Generate visualization charts")
    parser.add_argument("--log-to-mlflow", action="store_true",
                       help="Log charts and metrics to MLflow")
    parser.add_argument("--time-window", type=int, default=60,
                       help="Time window for Athena metrics (minutes)")

    args = parser.parse_args()

    # Test
    results = test_endpoint(
        endpoint_name=args.endpoint_name,
        num_samples=args.num_samples,
        data_source=args.data_source,
        test_data_path=args.test_data_path,
        enable_analytics=not args.disable_analytics,
        generate_charts=args.generate_charts,
        log_to_mlflow=args.log_to_mlflow,
        time_window_minutes=args.time_window,
    )

    print(f"\n✓ Testing completed!")
