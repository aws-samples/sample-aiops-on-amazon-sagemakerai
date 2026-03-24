"""
SageMaker Batch Transform for bulk fraud detection predictions.

Cost-effective batch processing with Athena input/output integration.

✅ MIGRATED: This file now uses PySpark for distributed data export.
- Uses AthenaClientPySpark for scalable processing
- Handles large datasets (1M+ rows) efficiently
- No awswrangler dependency for core operations

**Architecture:**
- Data export: PySpark (distributed)
- Batch transform: SageMaker native
- Results processing: Can use either PySpark or pandas depending on size
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

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
import mlflow

from src.config.config import (
    SAGEMAKER_EXEC_ROLE,
    BATCH_TRANSFORM_INSTANCE,
    BATCH_TRANSFORM_MAX_CONCURRENT,
    S3_BATCH_TRANSFORM_INPUT,
    S3_BATCH_TRANSFORM_OUTPUT,
    MLFLOW_BATCH_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    ATHENA_GROUND_TRUTH_TABLE,
    ATHENA_DATABASE,
)
from src.train_pipeline.athena.athena_client import AthenaClient
from src.utils.mlflow_utils import setup_mlflow_tracking, get_or_create_experiment
from src.utils.aws_utils import get_sagemaker_session, get_execution_role

logger = logging.getLogger(__name__)


def export_athena_to_s3(
    database: str,
    table_name: str,
    s3_path: str,
    filters: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """
    Export Athena table to S3 as JSON Lines using PySpark.

    Uses distributed PySpark processing for scalable data export.

    Args:
        database: Athena database name
        table_name: Athena table name
        s3_path: S3 destination path
        filters: Optional SQL WHERE clause
        limit: Optional row limit

    Returns:
        S3 path where data was written
    """
    logger.info(f"Exporting {table_name} to {s3_path} using PySpark")

    from src.train_pipeline.athena.athena_client_pyspark import AthenaClientPySpark

    # Use PySpark client for distributed export
    with AthenaClientPySpark(database=database) as client:
        # Read data from Athena
        df = client.read_table(table_name, filters=filters, limit=limit)

        # Check if data exists
        row_count = df.count()
        if row_count == 0:
            raise ValueError(f"No data found in {table_name}")

        logger.info(f"Exporting {row_count:,} rows to S3")

        # Drop columns not needed for inference
        columns_to_drop = ['is_fraud', 'fraud_prediction', 'fraud_probability',
                          'ground_truth', 'ground_truth_fraud', 'observed_fraud',
                          'data_version', 'created_at', 'updated_at',
                          'ingestion_timestamp', 'batch_id', 'data_source',
                          'transaction_id', 'customer_id']  # Also drop IDs

        # Drop columns that exist
        df = df.drop(*[col for col in columns_to_drop if col in df.columns])

        # Export to S3 as JSON Lines (SageMaker batch transform format)
        output_path = f"{s3_path.rstrip('/')}/input"

        # Write as JSON Lines using Spark (distributed write)
        logger.info(f"Writing JSON Lines to {output_path}")
        (df.coalesce(1)  # Single file for batch transform
         .write
         .mode("overwrite")
         .json(output_path))

        logger.info(f"✓ Data exported to {output_path}")
        return output_path


def create_batch_transform_job(
    model_name: str,
    input_s3_path: str,
    output_s3_path: str,
    instance_type: str,
    instance_count: int,
    max_concurrent_transforms: int,
    sagemaker_session,
    role: str,
    job_name: Optional[str] = None,
) -> str:
    """
    Create SageMaker batch transform job.

    Args:
        model_name: SageMaker model name
        input_s3_path: S3 path to input data
        output_s3_path: S3 path for output
        instance_type: EC2 instance type
        instance_count: Number of instances
        max_concurrent_transforms: Max concurrent requests per instance
        sagemaker_session: SageMaker session
        role: IAM role ARN
        job_name: Optional job name

    Returns:
        Batch transform job name
    """
    if job_name is None:
        job_name = f"fraud-batch-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    logger.info(f"Creating batch transform job: {job_name}")

    sm_client = sagemaker_session.sagemaker_client

    # Create batch transform job
    sm_client.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        MaxConcurrentTransforms=max_concurrent_transforms,
        MaxPayloadInMB=6,  # 6 MB max
        BatchStrategy='MultiRecord',
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_s3_path
                }
            },
            'ContentType': 'application/jsonlines',
            'SplitType': 'Line',
        },
        TransformOutput={
            'S3OutputPath': output_s3_path,
            'AssembleWith': 'Line',
        },
        TransformResources={
            'InstanceType': instance_type,
            'InstanceCount': instance_count,
        },
    )

    logger.info(f"Batch transform job created: {job_name}")
    return job_name


def monitor_batch_job(
    job_name: str,
    sagemaker_session,
    check_interval: int = 30,
) -> Dict[str, Any]:
    """
    Monitor batch transform job until completion.

    Args:
        job_name: Batch transform job name
        sagemaker_session: SageMaker session
        check_interval: Seconds between status checks

    Returns:
        Job description dictionary
    """
    sm_client = sagemaker_session.sagemaker_client

    logger.info(f"Monitoring batch transform job: {job_name}")

    while True:
        response = sm_client.describe_transform_job(TransformJobName=job_name)
        status = response['TransformJobStatus']

        logger.info(f"Job status: {status}")

        if status in ['Completed', 'Failed', 'Stopped']:
            break

        time.sleep(check_interval)

    if status == 'Completed':
        logger.info(f"✓ Batch transform job completed successfully")
    else:
        failure_reason = response.get('FailureReason', 'Unknown')
        logger.error(f"✗ Batch transform job {status}: {failure_reason}")

    return response


def process_batch_results(
    output_s3_path: str,
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process batch transform results from S3.

    Args:
        output_s3_path: S3 path to output files
        input_df: Original input DataFrame

    Returns:
        DataFrame with predictions merged with input
    """
    import awswrangler as wr

    logger.info(f"Processing batch results from {output_s3_path}")

    # Read all output files from S3
    output_files = wr.s3.list_objects(output_s3_path)

    if not output_files:
        raise ValueError(f"No output files found in {output_s3_path}")

    # Read predictions
    predictions_list = []
    for file_path in output_files:
        if file_path.endswith('.out'):
            # Read JSON lines
            df_pred = wr.s3.read_json(file_path, lines=True)
            predictions_list.append(df_pred)

    if not predictions_list:
        raise ValueError("No prediction files found")

    # Combine all predictions
    predictions_df = pd.concat(predictions_list, ignore_index=True)

    logger.info(f"Processed {len(predictions_df)} predictions")

    # Merge predictions with input data
    # Assuming predictions are in same order as input
    if len(predictions_df) != len(input_df):
        logger.warning(
            f"Prediction count ({len(predictions_df)}) "
            f"doesn't match input count ({len(input_df)})"
        )

    # Extract predictions from response format
    # SageMaker returns: {"predictions": [0], "probabilities": {...}}
    if 'predictions' in predictions_df.columns:
        input_df['prediction'] = predictions_df['predictions'].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )
    if 'probabilities' in predictions_df.columns:
        input_df['probability_fraud'] = predictions_df['probabilities'].apply(
            lambda x: x.get('fraud', [0])[0] if isinstance(x, dict) else 0
        )
        input_df['probability_non_fraud'] = predictions_df['probabilities'].apply(
            lambda x: x.get('non_fraud', [0])[0] if isinstance(x, dict) else 0
        )

    return input_df


def write_results_to_athena(
    athena_client: AthenaClient,
    results_df: pd.DataFrame,
    endpoint_name: str,
    model_version: str,
    mlflow_run_id: str,
) -> int:
    """
    Write batch transform results to Athena inference_responses table.

    Args:
        athena_client: AthenaClient instance
        results_df: DataFrame with predictions
        endpoint_name: Endpoint name for tracking
        model_version: Model version
        mlflow_run_id: MLflow run ID

    Returns:
        Number of rows written
    """
    logger.info("Writing batch results to Athena")

    # Add metadata columns
    results_df['inference_id'] = [str(uuid.uuid4()) for _ in range(len(results_df))]
    results_df['request_timestamp'] = datetime.utcnow()
    results_df['endpoint_name'] = f"{endpoint_name}-batch"
    results_df['model_version'] = model_version
    results_df['mlflow_run_id'] = mlflow_run_id
    results_df['inference_mode'] = 'batch'

    # Calculate confidence scores
    results_df['confidence_score'] = results_df[['probability_fraud', 'probability_non_fraud']].max(axis=1)
    results_df['is_high_confidence'] = results_df['confidence_score'] > 0.9
    results_df['is_low_confidence'] = (results_df['confidence_score'] >= 0.4) & (results_df['confidence_score'] <= 0.6)

    # Convert input features to JSON
    feature_cols = [col for col in results_df.columns
                   if col not in ['prediction', 'probability_fraud', 'probability_non_fraud',
                                 'inference_id', 'request_timestamp', 'endpoint_name',
                                 'model_version', 'mlflow_run_id', 'inference_mode',
                                 'confidence_score', 'is_high_confidence', 'is_low_confidence']]

    results_df['input_features'] = results_df[feature_cols].apply(
        lambda row: json.dumps(row.to_dict()), axis=1
    )

    # Select columns for Athena
    athena_cols = [
        'inference_id', 'request_timestamp', 'endpoint_name', 'model_version',
        'mlflow_run_id', 'input_features', 'prediction', 'probability_fraud',
        'probability_non_fraud', 'confidence_score', 'is_high_confidence',
        'is_low_confidence', 'inference_mode'
    ]

    write_df = results_df[athena_cols].copy()

    # Write to Athena
    athena_client.write_dataframe(
        write_df,
        'inference_responses',
        mode='append'
    )

    logger.info(f"✓ Wrote {len(write_df)} batch predictions to Athena")
    return len(write_df)


def batch_transform(
    model_uri: str,
    input_s3_path: Optional[str] = None,
    input_athena_table: Optional[str] = None,
    athena_filter: Optional[str] = None,
    limit: Optional[int] = None,
    output_s3_path: Optional[str] = None,
    instance_type: str = BATCH_TRANSFORM_INSTANCE,
    instance_count: int = 1,
    max_concurrent: int = BATCH_TRANSFORM_MAX_CONCURRENT,
    write_to_athena: bool = True,
    generate_charts: bool = True,
    endpoint_name: str = "fraud-batch",
) -> Dict[str, Any]:
    """
    Run batch transform job with Athena integration.

    Args:
        model_uri: MLflow model URI (e.g., 'runs:/<run-id>/model')
        input_s3_path: S3 path to input data (mutually exclusive with input_athena_table)
        input_athena_table: Athena table name for input
        athena_filter: SQL WHERE clause for filtering Athena data
        limit: Row limit for Athena query
        output_s3_path: S3 path for output (default: config)
        instance_type: EC2 instance type
        instance_count: Number of instances
        max_concurrent: Max concurrent transforms per instance
        write_to_athena: Write results to Athena
        generate_charts: Generate analytics charts
        endpoint_name: Name for tracking

    Returns:
        Dictionary with batch job results and metrics
    """
    print("=" * 80)
    print("SAGEMAKER BATCH TRANSFORM")
    print("=" * 80)
    print(f"Model: {model_uri}")
    print(f"Instance: {instance_type} (count={instance_count})")
    print(f"Write to Athena: {write_to_athena}")
    print("=" * 80 + "\n")

    # Setup MLflow
    setup_mlflow_tracking(MLFLOW_TRACKING_URI)
    experiment_id = get_or_create_experiment(MLFLOW_BATCH_EXPERIMENT_NAME)

    # Initialize clients
    athena_client = AthenaClient()
    sagemaker_session = get_sagemaker_session()

    # Use config role if available
    if SAGEMAKER_EXEC_ROLE:
        role = SAGEMAKER_EXEC_ROLE
    else:
        role = get_execution_role(sagemaker_session=sagemaker_session)

    # Default S3 paths
    if output_s3_path is None:
        output_s3_path = S3_BATCH_TRANSFORM_OUTPUT

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("instance_type", instance_type)
        mlflow.log_param("instance_count", instance_count)
        mlflow.log_param("endpoint_name", endpoint_name)

        # Step 1: Prepare input data
        if input_athena_table:
            logger.info(f"Reading input from Athena table: {input_athena_table}")

            # Export Athena table to S3
            input_s3_path = f"{S3_BATCH_TRANSFORM_INPUT}{run_id}/"
            input_file = export_athena_to_s3(
                athena_client,
                input_athena_table,
                input_s3_path,
                filters=athena_filter,
                limit=limit,
            )

            # Also keep DataFrame for later merging
            input_df = athena_client.read_table(
                input_athena_table,
                filters=athena_filter,
                limit=limit
            )

            mlflow.log_param("input_source", "athena")
            mlflow.log_param("input_table", input_athena_table)
            mlflow.log_metric("input_row_count", len(input_df))

        elif input_s3_path:
            logger.info(f"Using input from S3: {input_s3_path}")
            # Load input data for later merging
            import awswrangler as wr
            input_df = wr.s3.read_json(input_s3_path, lines=True)

            mlflow.log_param("input_source", "s3")
            mlflow.log_param("input_path", input_s3_path)
            mlflow.log_metric("input_row_count", len(input_df))

        else:
            raise ValueError("Must specify either input_s3_path or input_athena_table")

        # Step 2: Get or create SageMaker model from MLflow
        # For batch transform, we need a deployed model
        # This is simplified - in practice you'd deploy the model first
        logger.info("Note: Batch transform requires a deployed SageMaker model")
        logger.info("      Use deploy.py first, then reference that model")

        # Extract run_id from model_uri
        if model_uri.startswith("runs:/"):
            mlflow_run_id = model_uri.split("/")[1]
            model_version = mlflow_run_id[:8]
        else:
            mlflow_run_id = "unknown"
            model_version = "unknown"

        # For now, assume model is already deployed
        # In production, you'd either:
        # 1. Deploy the model specifically for batch transform
        # 2. Use an existing endpoint's model
        model_name = f"fraud-detector-{model_version}"

        # Step 3: Create and run batch transform job
        output_path = f"{output_s3_path}{run_id}/"

        job_name = create_batch_transform_job(
            model_name=model_name,
            input_s3_path=input_s3_path or input_file,
            output_s3_path=output_path,
            instance_type=instance_type,
            instance_count=instance_count,
            max_concurrent_transforms=max_concurrent,
            sagemaker_session=sagemaker_session,
            role=role,
        )

        mlflow.log_param("batch_job_name", job_name)

        # Step 4: Monitor job
        start_time = time.time()
        job_info = monitor_batch_job(job_name, sagemaker_session)
        duration = time.time() - start_time

        mlflow.log_metric("batch_duration_seconds", duration)
        mlflow.log_metric("batch_duration_minutes", duration / 60)

        if job_info['TransformJobStatus'] != 'Completed':
            failure_reason = job_info.get('FailureReason', 'Unknown')
            mlflow.log_param("failure_reason", failure_reason)
            raise RuntimeError(f"Batch transform failed: {failure_reason}")

        # Step 5: Process results
        results_df = process_batch_results(output_path, input_df)

        # Calculate metrics
        total_predictions = len(results_df)
        fraud_predictions = (results_df['prediction'] == 1).sum()
        avg_fraud_prob = results_df['probability_fraud'].mean()

        mlflow.log_metric("total_predictions", total_predictions)
        mlflow.log_metric("fraud_predictions", fraud_predictions)
        mlflow.log_metric("fraud_rate", fraud_predictions / total_predictions)
        mlflow.log_metric("avg_fraud_probability", avg_fraud_prob)

        # Step 6: Write to Athena
        if write_to_athena:
            rows_written = write_results_to_athena(
                athena_client,
                results_df,
                endpoint_name,
                model_version,
                mlflow_run_id,
            )
            mlflow.log_metric("athena_rows_written", rows_written)

        # Step 7: Generate charts
        if generate_charts:
            try:
                from src.utils.visualization_utils import log_all_charts_to_mlflow

                logger.info("Generating analytics charts...")
                # Wait a bit for Athena to ingest the data
                time.sleep(10)

                chart_metrics = log_all_charts_to_mlflow(
                    athena_client,
                    f"{endpoint_name}-batch",
                    mlflow_run_id=run_id,
                    days=7,
                )
                logger.info("✓ Charts generated and logged to MLflow")

            except Exception as e:
                logger.error(f"Error generating charts: {e}")

        # Return summary
        summary = {
            'run_id': run_id,
            'job_name': job_name,
            'status': 'completed',
            'total_predictions': int(total_predictions),
            'fraud_predictions': int(fraud_predictions),
            'fraud_rate': float(fraud_predictions / total_predictions),
            'avg_fraud_probability': float(avg_fraud_prob),
            'duration_minutes': duration / 60,
            'output_s3_path': output_path,
        }

        logger.info("\n" + "=" * 80)
        logger.info("BATCH TRANSFORM COMPLETED")
        logger.info("=" * 80)
        for key, value in summary.items():
            logger.info(f"{key}: {value}")

        return summary


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run batch transform")
    parser.add_argument("--model-uri", required=True,
                       help="MLflow model URI (e.g., runs:/<run-id>/model)")
    parser.add_argument("--input-athena-table",
                       help="Athena table for input data")
    parser.add_argument("--input-s3-path",
                       help="S3 path to input data")
    parser.add_argument("--athena-filter",
                       help="SQL WHERE clause for Athena filtering")
    parser.add_argument("--limit", type=int,
                       help="Row limit for Athena query")
    parser.add_argument("--output-s3-path",
                       help="S3 path for output")
    parser.add_argument("--instance-type", default=BATCH_TRANSFORM_INSTANCE)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=BATCH_TRANSFORM_MAX_CONCURRENT)
    parser.add_argument("--no-athena", action="store_true",
                       help="Don't write results to Athena")
    parser.add_argument("--no-charts", action="store_true",
                       help="Don't generate charts")
    parser.add_argument("--endpoint-name", default="fraud-batch")

    args = parser.parse_args()

    # Run batch transform
    results = batch_transform(
        model_uri=args.model_uri,
        input_s3_path=args.input_s3_path,
        input_athena_table=args.input_athena_table,
        athena_filter=args.athena_filter,
        limit=args.limit,
        output_s3_path=args.output_s3_path,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        max_concurrent=args.max_concurrent,
        write_to_athena=not args.no_athena,
        generate_charts=not args.no_charts,
        endpoint_name=args.endpoint_name,
    )

    print(f"\n✓ Batch transform completed successfully!")
    print(f"  Run ID: {results['run_id']}")
    print(f"  Predictions: {results['total_predictions']}")
    print(f"  Fraud Rate: {results['fraud_rate']:.2%}")
