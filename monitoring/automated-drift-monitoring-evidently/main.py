#!/usr/bin/env python3
"""
SageMaker-focused CLI for fraud detection pipeline with Athena integration.

This CLI provides unified access to:
- Training (local or SageMaker)
- Deployment (with Athena logging)
- Testing (with analytics)
- Infrastructure setup

Usage:
    python main.py setup --migrate-data
    python main.py train --training-mode local --data-source athena
    python main.py deploy --run-id <run-id> --endpoint-name fraud-detector
    python main.py test --endpoint-name fraud-detector --num-samples 50
"""

import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables BEFORE any other imports
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_command(args):
    """Run infrastructure setup."""
    from src.setup.setup_athena_tables import main as setup_main

    # Override sys.argv for setup script
    sys.argv = ['setup_athena_tables.py']
    if args.migrate_data:
        sys.argv.append('--migrate-data')
    if args.verify_only:
        sys.argv.append('--verify-only')
    if args.skip_s3:
        sys.argv.append('--skip-s3')
    if args.region:
        sys.argv.extend(['--region', args.region])

    return setup_main()


def setup_logging_command(args):
    """Setup SQS queue and Lambda consumer for inference logging."""
    from src.setup.setup_inference_logging import setup_sqs_queue, setup_lambda_consumer
    from src.config.config import SQS_QUEUE_NAME, SAGEMAKER_EXEC_ROLE, AWS_DEFAULT_REGION
    import boto3

    account_id = boto3.client('sts').get_caller_identity()['Account']

    logger.info(f"Setting up inference logging infrastructure...")
    queue_url = setup_sqs_queue()
    queue_arn = f"arn:aws:sqs:{AWS_DEFAULT_REGION}:{account_id}:{SQS_QUEUE_NAME}"
    setup_lambda_consumer(SAGEMAKER_EXEC_ROLE, queue_arn)

    print(f"\n✓ Inference logging setup complete!")
    print(f"  SQS Queue URL: {queue_url}")
    print(f"  Lambda consumer created with SQS trigger")

    return 0


def train_command(args):
    """Run training."""
    from src.pipeline.train import train

    logger.info("Starting training...")

    run_id, model = train(
        training_mode=args.training_mode,
        data_source=args.data_source,
        data_path=args.data_path,
        athena_table=args.athena_table,
        athena_filter=args.athena_filter,
        instance_type=args.instance_type,
        experiment_name=args.experiment_name,
    )

    print(f"\n✓ Training completed successfully!")
    print(f"  Run ID: {run_id}")
    print(f"  Model saved to MLflow")

    return 0


def deploy_command(args):
    """Run deployment."""
    from src.pipeline.deploy import deploy

    logger.info("Starting deployment...")

    predictor = deploy(
        run_id=args.run_id,
        endpoint_name=args.endpoint_name,
        model_version=args.model_version,
        memory_size_mb=args.memory_size,
        max_concurrency=args.max_concurrency,
        enable_athena_logging=args.enable_athena_logging,
    )

    print(f"\n✓ Deployment completed successfully!")
    print(f"  Endpoint: {args.endpoint_name}")
    print(f"  Status: InService")

    return 0


def test_command(args):
    """Run endpoint testing."""
    from src.pipeline.test_endpoint import test_endpoint

    logger.info("Starting endpoint testing...")

    results = test_endpoint(
        endpoint_name=args.endpoint_name,
        num_samples=args.num_samples,
        data_source=args.data_source,
        test_data_path=args.test_data_path,
        enable_analytics=args.enable_analytics,
        generate_charts=args.generate_charts,
        log_to_mlflow=args.log_to_mlflow,
        time_window_minutes=args.time_window,
    )

    print(f"\n✓ Testing completed successfully!")

    return 0


def batch_command(args):
    """Run batch transform."""
    from src.pipeline.batch_transform import batch_transform

    logger.info("Starting batch transform...")

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
        write_to_athena=args.write_to_athena,
        generate_charts=args.generate_charts,
        endpoint_name=args.endpoint_name,
    )

    print(f"\n✓ Batch transform completed successfully!")
    print(f"  Predictions: {results['total_predictions']}")
    print(f"  Fraud Rate: {results['fraud_rate']:.2%}")

    return 0


def pipeline_command(args):
    """Run SageMaker Pipeline operations."""
    from src.pipeline.pipeline_cli import (
        create_pipeline, start_execution, list_executions,
        describe_execution, list_pipeline_versions, delete_pipeline
    )

    if args.pipeline_action == 'create':
        result = create_pipeline(
            pipeline_name=args.pipeline_name,
            region=args.region,
            role=args.role,
            include_deployment=not args.no_deployment
        )
        print(json.dumps(result, indent=2))

    elif args.pipeline_action == 'start':
        result = start_execution(
            pipeline_name=args.pipeline_name,
            execution_name=args.execution_name,
            parameters=args.parameters,
            wait=args.wait
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.pipeline_action == 'list':
        results = list_executions(
            pipeline_name=args.pipeline_name,
            max_results=args.max_results
        )
        print(json.dumps(results, indent=2))

    elif args.pipeline_action == 'describe':
        result = describe_execution(args.execution_arn)
        print(json.dumps(result, indent=2))

    elif args.pipeline_action == 'versions':
        results = list_pipeline_versions(
            pipeline_name=args.pipeline_name,
            max_results=args.max_results
        )
        print(json.dumps(results, indent=2))

    elif args.pipeline_action == 'delete':
        if not args.confirm:
            print("WARNING: This will delete the pipeline permanently.")
            print("Add --confirm to proceed.")
            return 1
        result = delete_pipeline(args.pipeline_name)
        print(json.dumps(result, indent=2))

    return 0

def full_command(args):
    """Run full pipeline (train → deploy → test)."""
    logger.info("Starting full pipeline...")

    # Step 1: Train
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING")
    print("=" * 80 + "\n")

    from src.pipeline.train import train

    run_id, model = train(
        training_mode=args.training_mode,
        data_source=args.data_source,
        instance_type=args.instance_type,
    )

    # Step 2: Deploy
    print("\n" + "=" * 80)
    print("STEP 2: DEPLOYMENT")
    print("=" * 80 + "\n")

    from src.pipeline.deploy import deploy

    predictor = deploy(
        run_id=run_id,
        endpoint_name=args.endpoint_name,
        enable_athena_logging=True,
    )

    # Step 3: Test
    print("\n" + "=" * 80)
    print("STEP 3: TESTING")
    print("=" * 80 + "\n")

    from src.pipeline.test_endpoint import test_endpoint

    results = test_endpoint(
        endpoint_name=args.endpoint_name,
        num_samples=args.num_samples,
        enable_analytics=True,
    )

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nRun ID: {run_id}")
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Predictions: {results['realtime']['total_predictions']}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SageMaker Fraud Detection Pipeline with Athena Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup infrastructure
  python main.py setup --migrate-data

  # Train model
  python main.py train --training-mode local --data-source athena

  # Deploy model
  python main.py deploy --run-id abc123 --endpoint-name fraud-detector

  # Test endpoint
  python main.py test --endpoint-name fraud-detector --num-samples 50

  # Run full pipeline
  python main.py full --endpoint-name fraud-detector
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # =========================================================================
    # Setup Command
    # =========================================================================
    setup_parser = subparsers.add_parser('setup', help='Setup Athena infrastructure')
    setup_parser.add_argument(
        '--migrate-data',
        action='store_true',
        help='Migrate CSV data to Athena tables'
    )
    setup_parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing setup'
    )
    setup_parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    setup_parser.add_argument(
        '--skip-s3',
        action='store_true',
        help='Skip S3 bucket creation'
    )

    # =========================================================================
    # Setup Logging Command
    # =========================================================================
    subparsers.add_parser('setup-logging', help='Setup SQS queue and Lambda for inference logging')

    # =========================================================================
    # Train Command
    # =========================================================================
    train_parser = subparsers.add_parser('train', help='Train fraud detection model')
    train_parser.add_argument(
        '--training-mode',
        choices=['local', 'sagemaker'],
        default='local',
        help='Training mode (default: local)'
    )
    train_parser.add_argument(
        '--data-source',
        choices=['athena', 'csv'],
        default='athena',
        help='Data source (default: athena)'
    )
    train_parser.add_argument(
        '--data-path',
        help='Path to CSV file (if data-source=csv)'
    )
    train_parser.add_argument(
        '--athena-table',
        default='training_data',
        help='Athena table name (default: training_data)'
    )
    train_parser.add_argument(
        '--athena-filter',
        help='SQL WHERE clause for filtering Athena data'
    )
    train_parser.add_argument(
        '--instance-type',
        default='ml.m5.xlarge',
        help='SageMaker instance type (for sagemaker mode)'
    )
    train_parser.add_argument(
        '--experiment-name',
        help='MLflow experiment name'
    )

    # =========================================================================
    # Deploy Command
    # =========================================================================
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model to SageMaker')
    deploy_parser.add_argument(
        '--run-id',
        required=True,
        help='MLflow run ID'
    )
    deploy_parser.add_argument(
        '--endpoint-name',
        required=True,
        help='SageMaker endpoint name'
    )
    deploy_parser.add_argument(
        '--model-version',
        help='Model version tag'
    )
    deploy_parser.add_argument(
        '--memory-size',
        type=int,
        help='Memory size in MB (1024-6144)'
    )
    deploy_parser.add_argument(
        '--max-concurrency',
        type=int,
        help='Max concurrent invocations (1-200)'
    )
    deploy_parser.add_argument(
        '--enable-athena-logging',
        action='store_true',
        default=True,
        help='Enable Athena logging (default: true)'
    )

    # =========================================================================
    # Test Command
    # =========================================================================
    test_parser = subparsers.add_parser('test', help='Test SageMaker endpoint')
    test_parser.add_argument(
        '--endpoint-name',
        required=True,
        help='SageMaker endpoint name'
    )
    test_parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of test samples (default: 100)'
    )
    test_parser.add_argument(
        '--data-source',
        choices=['csv', 'athena'],
        default='csv',
        help='Test data source (default: csv)'
    )
    test_parser.add_argument(
        '--test-data-path',
        help='Path to CSV test data'
    )
    test_parser.add_argument(
        '--enable-analytics',
        action='store_true',
        default=True,
        help='Query Athena for analytics (default: true)'
    )
    test_parser.add_argument(
        '--generate-charts',
        action='store_true',
        help='Generate visualization charts'
    )
    test_parser.add_argument(
        '--log-to-mlflow',
        action='store_true',
        help='Log charts and metrics to MLflow'
    )
    test_parser.add_argument(
        '--time-window',
        type=int,
        default=60,
        help='Time window for Athena metrics in minutes (default: 60)'
    )

    # =========================================================================
    # Batch Command
    # =========================================================================
    batch_parser = subparsers.add_parser('batch', help='Run batch transform')
    batch_parser.add_argument(
        '--model-uri',
        required=True,
        help='MLflow model URI (e.g., runs:/<run-id>/model)'
    )
    batch_parser.add_argument(
        '--input-athena-table',
        help='Athena table for input data'
    )
    batch_parser.add_argument(
        '--input-s3-path',
        help='S3 path to input data'
    )
    batch_parser.add_argument(
        '--athena-filter',
        help='SQL WHERE clause for Athena filtering'
    )
    batch_parser.add_argument(
        '--limit',
        type=int,
        help='Row limit for Athena query'
    )
    batch_parser.add_argument(
        '--output-s3-path',
        help='S3 path for output'
    )
    batch_parser.add_argument(
        '--instance-type',
        default='ml.m5.xlarge',
        help='EC2 instance type (default: ml.m5.xlarge)'
    )
    batch_parser.add_argument(
        '--instance-count',
        type=int,
        default=1,
        help='Number of instances (default: 1)'
    )
    batch_parser.add_argument(
        '--max-concurrent',
        type=int,
        default=4,
        help='Max concurrent transforms per instance (default: 4)'
    )
    batch_parser.add_argument(
        '--write-to-athena',
        action='store_true',
        default=True,
        help='Write results to Athena (default: true)'
    )
    batch_parser.add_argument(
        '--generate-charts',
        action='store_true',
        default=True,
        help='Generate analytics charts (default: true)'
    )
    batch_parser.add_argument(
        '--endpoint-name',
        default='fraud-batch',
        help='Name for tracking (default: fraud-batch)'
    )

    # =========================================================================
    # Pipeline Command
    # =========================================================================
    pipeline_parser = subparsers.add_parser('pipeline', help='Manage SageMaker Pipelines')
    pipeline_parser.add_argument(
        'pipeline_action',
        choices=['create', 'start', 'list', 'describe', 'versions', 'delete'],
        help='Pipeline action'
    )
    pipeline_parser.add_argument(
        '--pipeline-name',
        default='fraud-detection-pipeline',
        help='Pipeline name (default: fraud-detection-pipeline)'
    )
    pipeline_parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    pipeline_parser.add_argument(
        '--role',
        help='SageMaker execution role ARN'
    )
    pipeline_parser.add_argument(
        '--execution-name',
        help='Execution name (for start action)'
    )
    pipeline_parser.add_argument(
        '--execution-arn',
        help='Execution ARN (for describe action)'
    )
    pipeline_parser.add_argument(
        '--parameters',
        type=json.loads,
        help='Pipeline parameters as JSON string (for start action)'
    )
    pipeline_parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for execution to complete (for start action)'
    )
    pipeline_parser.add_argument(
        '--max-results',
        type=int,
        default=10,
        help='Maximum number of results (for list/versions actions)'
    )
    pipeline_parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm deletion (for delete action)'
    )
    pipeline_parser.add_argument(
        '--no-deployment',
        action='store_true',
        help='Exclude deployment and testing steps from pipeline (for create action)'
    )

    # =========================================================================
    # Full Command
    # =========================================================================
    full_parser = subparsers.add_parser('full', help='Run full pipeline (train → deploy → test)')
    full_parser.add_argument(
        '--endpoint-name',
        required=True,
        help='SageMaker endpoint name'
    )
    full_parser.add_argument(
        '--training-mode',
        choices=['local', 'sagemaker'],
        default='local',
        help='Training mode (default: local)'
    )
    full_parser.add_argument(
        '--data-source',
        choices=['athena', 'csv'],
        default='athena',
        help='Data source (default: athena)'
    )
    full_parser.add_argument(
        '--instance-type',
        default='ml.m5.xlarge',
        help='SageMaker instance type'
    )
    full_parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of test samples (default: 50)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        if args.command == 'setup':
            return setup_command(args)
        elif args.command == 'setup-logging':
            return setup_logging_command(args)
        elif args.command == 'train':
            return train_command(args)
        elif args.command == 'deploy':
            return deploy_command(args)
        elif args.command == 'test':
            return test_command(args)
        elif args.command == 'batch':
            return batch_command(args)
        elif args.command == 'pipeline':
            return pipeline_command(args)
        elif args.command == 'full':
            return full_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
