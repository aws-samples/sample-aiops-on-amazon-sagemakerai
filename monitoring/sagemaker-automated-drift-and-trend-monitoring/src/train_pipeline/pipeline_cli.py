"""
CLI for SageMaker Pipeline operations.

This CLI provides commands for:
- Creating/updating pipelines (with optional deployment steps)
- Starting pipeline executions
- Listing executions
- Describing execution details
- Listing pipeline versions
- Deleting pipelines
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import boto3
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize boto3 clients
sagemaker_client = boto3.client('sagemaker')


def create_pipeline(
    pipeline_name: str,
    region: str = "us-east-1",
    role: Optional[str] = None,
    description: str = "Fraud detection pipeline with MLflow monitoring",
    include_deployment: bool = True,
    tags: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Create or update SageMaker Pipeline.

    Args:
        pipeline_name: Pipeline name
        region: AWS region
        role: SageMaker execution role
        description: Pipeline description
        include_deployment: Include deployment and testing steps
        tags: Optional tags

    Returns:
        Dictionary with pipeline ARN
    """
    logger.info(f"Creating pipeline: {pipeline_name}")
    logger.info(f"  Include deployment: {include_deployment}")

    try:
        # Import pipeline builder
        from src.train_pipeline.pipeline import FraudDetectionPipeline

        # Create pipeline
        pipeline_builder = FraudDetectionPipeline(
            pipeline_name=pipeline_name,
            region=region,
            role=role
        )

        # Upsert pipeline
        pipeline = pipeline_builder.create_pipeline(
            include_deployment=include_deployment,
        )

        result = {
            'pipeline_arn': pipeline.name,
            'pipeline_name': pipeline_name,
            'status': 'created',
        }

        logger.info(f"✓ Pipeline created: {result['pipeline_arn']}")
        return result

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}", exc_info=True)
        raise

def start_execution(
    pipeline_name: str,
    execution_name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    wait: bool = False
) -> Dict[str, Any]:
    """
    Start pipeline execution.

    Args:
        pipeline_name: Pipeline name
        execution_name: Optional execution name (auto-generated if not provided)
        parameters: Pipeline parameters
        wait: Wait for execution to complete

    Returns:
        Dictionary with execution ARN and status
    """
    # Generate execution name if not provided
    if execution_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        execution_name = f"{pipeline_name}-{timestamp}"

    logger.info(f"Starting pipeline execution: {execution_name}")

    try:
        # Format parameters for SageMaker
        pipeline_params = []
        if parameters:
            for key, value in parameters.items():
                pipeline_params.append({
                    'Name': key,
                    'Value': str(value)
                })

        # Start execution
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineExecutionDisplayName=execution_name,
            PipelineParameters=pipeline_params
        )

        execution_arn = response['PipelineExecutionArn']
        logger.info(f"✓ Execution started: {execution_arn}")

        result = {
            'execution_arn': execution_arn,
            'execution_name': execution_name,
            'pipeline_name': pipeline_name,
            'status': 'Executing',
            'parameters': parameters
        }

        # Wait for completion if requested
        if wait:
            logger.info("Waiting for execution to complete...")
            final_status = wait_for_execution(execution_arn)
            result['status'] = final_status

        return result

    except ClientError as e:
        logger.error(f"Failed to start execution: {e}")
        raise


def wait_for_execution(
    execution_arn: str,
    poll_interval: int = 30,
    max_wait_time: int = 3600
) -> str:
    """
    Wait for pipeline execution to complete.

    Args:
        execution_arn: Pipeline execution ARN
        poll_interval: Seconds between status checks
        max_wait_time: Maximum wait time in seconds

    Returns:
        Final execution status
    """
    start_time = time.time()

    while (time.time() - start_time) < max_wait_time:
        response = sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        status = response['PipelineExecutionStatus']
        logger.info(f"Execution status: {status}")

        if status in ['Succeeded', 'Failed', 'Stopped']:
            logger.info(f"✓ Execution completed with status: {status}")
            return status

        time.sleep(poll_interval)

    logger.warning(f"Execution still running after {max_wait_time}s")
    return 'Executing'


def list_executions(
    pipeline_name: str,
    max_results: int = 10,
    sort_by: str = 'CreationTime',
    sort_order: str = 'Descending'
) -> List[Dict[str, Any]]:
    """
    List pipeline executions.

    Args:
        pipeline_name: Pipeline name
        max_results: Maximum number of results
        sort_by: Sort by field (CreationTime or PipelineExecutionArn)
        sort_order: Sort order (Ascending or Descending)

    Returns:
        List of execution summaries
    """
    logger.info(f"Listing executions for pipeline: {pipeline_name}")

    try:
        response = sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline_name,
            MaxResults=max_results,
            SortBy=sort_by,
            SortOrder=sort_order
        )

        executions = response.get('PipelineExecutionSummaries', [])

        # Format results
        results = []
        for execution in executions:
            results.append({
                'execution_arn': execution['PipelineExecutionArn'],
                'execution_name': execution.get('PipelineExecutionDisplayName', 'N/A'),
                'status': execution['PipelineExecutionStatus'],
                'start_time': execution['StartTime'].isoformat(),
                'end_time': execution.get('EndTime', datetime.now()).isoformat() if execution.get('EndTime') else None
            })

        logger.info(f"✓ Found {len(results)} executions")
        return results

    except ClientError as e:
        logger.error(f"Failed to list executions: {e}")
        raise


def describe_execution(execution_arn: str) -> Dict[str, Any]:
    """
    Describe pipeline execution with step details.

    Args:
        execution_arn: Pipeline execution ARN

    Returns:
        Dictionary with execution details
    """
    logger.info(f"Describing execution: {execution_arn}")

    try:
        # Get execution details
        exec_response = sagemaker_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        # Get step details
        steps_response = sagemaker_client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn
        )

        steps = []
        for step in steps_response.get('PipelineExecutionSteps', []):
            step_info = {
                'name': step['StepName'],
                'status': step['StepStatus'],
                'start_time': step.get('StartTime', datetime.now()).isoformat() if step.get('StartTime') else None,
                'end_time': step.get('EndTime', datetime.now()).isoformat() if step.get('EndTime') else None,
            }

            # Add failure reason if failed
            if step['StepStatus'] == 'Failed' and 'FailureReason' in step:
                step_info['failure_reason'] = step['FailureReason']

            steps.append(step_info)

        result = {
            'execution_arn': execution_arn,
            'pipeline_name': exec_response['PipelineName'],
            'status': exec_response['PipelineExecutionStatus'],
            'start_time': exec_response['CreationTime'].isoformat(),
            'end_time': exec_response.get('LastModifiedTime', datetime.now()).isoformat() if exec_response.get('LastModifiedTime') else None,
            'parameters': exec_response.get('PipelineParameters', []),
            'steps': steps
        }

        # Add failure reason if failed
        if exec_response['PipelineExecutionStatus'] == 'Failed':
            if 'FailureReason' in exec_response:
                result['failure_reason'] = exec_response['FailureReason']

        logger.info(f"✓ Execution details retrieved")
        logger.info(f"  Status: {result['status']}")
        logger.info(f"  Steps: {len(steps)}")

        return result

    except ClientError as e:
        logger.error(f"Failed to describe execution: {e}")
        raise


def list_pipeline_versions(
    pipeline_name: str,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    List pipeline versions.

    Args:
        pipeline_name: Pipeline name
        max_results: Maximum number of results

    Returns:
        List of pipeline versions
    """
    logger.info(f"Listing versions for pipeline: {pipeline_name}")

    try:
        response = sagemaker_client.list_pipeline_versions(
            PipelineName=pipeline_name,
            MaxResults=max_results
        )

        versions = []
        for version in response.get('PipelineVersionSummaries', []):
            versions.append({
                'version': version['PipelineVersion'],
                'created_time': version['CreationTime'].isoformat(),
                'status': version.get('PipelineStatus', 'Active')
            })

        logger.info(f"✓ Found {len(versions)} versions")
        return versions

    except ClientError as e:
        logger.error(f"Failed to list versions: {e}")
        raise


def delete_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """
    Delete pipeline.

    Args:
        pipeline_name: Pipeline name

    Returns:
        Dictionary with deletion status
    """
    logger.info(f"Deleting pipeline: {pipeline_name}")

    try:
        response = sagemaker_client.delete_pipeline(
            PipelineName=pipeline_name
        )

        result = {
            'pipeline_arn': response['PipelineArn'],
            'pipeline_name': pipeline_name,
            'status': 'deleted'
        }

        logger.info(f"✓ Pipeline deleted: {result['pipeline_arn']}")
        return result

    except ClientError as e:
        logger.error(f"Failed to delete pipeline: {e}")
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SageMaker Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create pipeline
  python pipeline_cli.py create-pipeline --pipeline-name fraud-detection-pipeline

  # Start execution
  python pipeline_cli.py start-execution --pipeline-name fraud-detection-pipeline

  # Start execution with parameters
  python pipeline_cli.py start-execution --pipeline-name fraud-detection-pipeline \\
    --parameters '{"EndpointName": "fraud-detector-v2", "MinRocAuc": "0.90"}'

  # List executions
  python pipeline_cli.py list-executions --pipeline-name fraud-detection-pipeline

  # Describe execution
  python pipeline_cli.py describe-execution --execution-arn <arn>

  # List versions
  python pipeline_cli.py list-versions --pipeline-name fraud-detection-pipeline

  # Delete pipeline
  python pipeline_cli.py delete-pipeline --pipeline-name fraud-detection-pipeline
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Create pipeline command
    create_parser = subparsers.add_parser('create-pipeline', help='Create or update pipeline')
    create_parser.add_argument('--pipeline-name', required=True, help='Pipeline name')
    create_parser.add_argument('--region', default='us-east-1', help='AWS region')
    create_parser.add_argument('--role', help='SageMaker execution role ARN')
    create_parser.add_argument('--description', default='Fraud detection pipeline',
                              help='Pipeline description')

    # Start execution command
    start_parser = subparsers.add_parser('start-execution', help='Start pipeline execution')
    start_parser.add_argument('--pipeline-name', required=True, help='Pipeline name')
    start_parser.add_argument('--execution-name', help='Execution name (auto-generated if not provided)')
    start_parser.add_argument('--parameters', type=json.loads,
                             help='Pipeline parameters as JSON string')
    start_parser.add_argument('--wait', action='store_true',
                             help='Wait for execution to complete')

    # List executions command
    list_exec_parser = subparsers.add_parser('list-executions', help='List pipeline executions')
    list_exec_parser.add_argument('--pipeline-name', required=True, help='Pipeline name')
    list_exec_parser.add_argument('--max-results', type=int, default=10,
                                  help='Maximum number of results')

    # Describe execution command
    describe_parser = subparsers.add_parser('describe-execution', help='Describe pipeline execution')
    describe_parser.add_argument('--execution-arn', required=True, help='Execution ARN')

    # List versions command
    versions_parser = subparsers.add_parser('list-versions', help='List pipeline versions')
    versions_parser.add_argument('--pipeline-name', required=True, help='Pipeline name')
    versions_parser.add_argument('--max-results', type=int, default=10,
                                help='Maximum number of results')

    # Delete pipeline command
    delete_parser = subparsers.add_parser('delete-pipeline', help='Delete pipeline')
    delete_parser.add_argument('--pipeline-name', required=True, help='Pipeline name')
    delete_parser.add_argument('--confirm', action='store_true',
                              help='Confirm deletion')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        # Execute command
        if args.command == 'create-pipeline':
            result = create_pipeline(
                pipeline_name=args.pipeline_name,
                region=args.region,
                role=args.role,
                description=args.description
            )
            print(json.dumps(result, indent=2))

        elif args.command == 'start-execution':
            result = start_execution(
                pipeline_name=args.pipeline_name,
                execution_name=args.execution_name,
                parameters=args.parameters,
                wait=args.wait
            )
            print(json.dumps(result, indent=2, default=str))

        elif args.command == 'list-executions':
            results = list_executions(
                pipeline_name=args.pipeline_name,
                max_results=args.max_results
            )
            print(json.dumps(results, indent=2))

        elif args.command == 'describe-execution':
            result = describe_execution(args.execution_arn)
            print(json.dumps(result, indent=2))

        elif args.command == 'list-versions':
            results = list_pipeline_versions(
                pipeline_name=args.pipeline_name,
                max_results=args.max_results
            )
            print(json.dumps(results, indent=2))

        elif args.command == 'delete-pipeline':
            if not args.confirm:
                print("WARNING: This will delete the pipeline permanently.")
                print("Use --confirm to proceed with deletion.")
                return 1

            result = delete_pipeline(args.pipeline_name)
            print(json.dumps(result, indent=2))

        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
