#!/usr/bin/env python3
"""
Create or Update SageMaker Execution Role with all required policies.

This script creates a new IAM role or updates an existing one with policies for:
- SageMaker (training, processing, endpoints)
- Lambda (for pipeline deployment steps)
- SageMaker Pipelines
- Athena/Glue access for data queries

If SAGEMAKER_EXEC_ROLE is set in .env, it will update that existing role.
Otherwise, it creates a new role with the specified name.

Usage:
    # Update existing role from .env
    python scripts/create_or_update_sagemaker_role.py
    
    # Create a new role
    python scripts/create_or_update_sagemaker_role.py --role-name my-sagemaker-role
    
    # Force update existing role by ARN
    python scripts/create_or_update_sagemaker_role.py --role-arn arn:aws:iam::123456789:role/MyRole
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Trust policy allowing SageMaker and Lambda to assume the role
TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}

# AWS Managed policies to attach
MANAGED_POLICIES = [
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    "arn:aws:iam::aws:policy/AmazonAthenaFullAccess",
    "arn:aws:iam::aws:policy/AWSGlueConsoleFullAccess",
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
]


def get_custom_policy(account_id: str, region: str) -> dict:
    """
    Get custom inline policy for additional permissions.
    
    Args:
        account_id: AWS account ID
        region: AWS region
        
    Returns:
        Policy document
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "MLflowAccess",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:CreateMlflowTrackingServer",
                    "sagemaker:UpdateMlflowTrackingServer",
                    "sagemaker:DeleteMlflowTrackingServer",
                    "sagemaker:DescribeMlflowTrackingServer",
                    "sagemaker:StartMlflowTrackingServer",
                    "sagemaker:StopMlflowTrackingServer"
                ],
                "Resource": f"arn:aws:sagemaker:{region}:{account_id}:mlflow-tracking-server/*"
            },
            {
                "Sid": "PassRoleToSageMaker",
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": f"arn:aws:iam::{account_id}:role/*",
                "Condition": {
                    "StringEquals": {
                        "iam:PassedToService": [
                            "sagemaker.amazonaws.com",
                            "lambda.amazonaws.com"
                        ]
                    }
                }
            },
            {
                "Sid": "LambdaInvoke",
                "Effect": "Allow",
                "Action": [
                    "lambda:InvokeFunction",
                    "lambda:CreateFunction",
                    "lambda:DeleteFunction",
                    "lambda:UpdateFunctionCode",
                    "lambda:UpdateFunctionConfiguration",
                    "lambda:GetFunction",
                    "lambda:ListFunctions",
                    "lambda:CreateEventSourceMapping",
                    "lambda:DeleteEventSourceMapping",
                    "lambda:GetEventSourceMapping",
                    "lambda:ListEventSourceMappings",
                    "lambda:UpdateEventSourceMapping"
                ],
                "Resource": "*"
            },
            {
                "Sid": "SQSAccess",
                "Effect": "Allow",
                "Action": [
                    "sqs:CreateQueue",
                    "sqs:DeleteQueue",
                    "sqs:GetQueueUrl",
                    "sqs:GetQueueAttributes",
                    "sqs:SetQueueAttributes",
                    "sqs:SendMessage",
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:PurgeQueue",
                    "sqs:ListQueues"
                ],
                "Resource": "*"
            },
            {
                "Sid": "ECRAccess",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage"
                ],
                "Resource": "*"
            },
            {
                "Sid": "KMSAccess",
                "Effect": "Allow",
                "Action": [
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:GenerateDataKey"
                ],
                "Resource": "*"
            },
            {
                "Sid": "GlueFullAccess",
                "Effect": "Allow",
                "Action": [
                    "glue:GetDatabase",
                    "glue:GetDatabases",
                    "glue:GetTable",
                    "glue:GetTables",
                    "glue:GetPartition",
                    "glue:GetPartitions",
                    "glue:BatchGetPartition",
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:DeleteTable",
                    "glue:CreateDatabase",
                    "glue:GetCatalogImportStatus"
                ],
                "Resource": "*"
            },
            {
                "Sid": "AthenaFullAccess",
                "Effect": "Allow",
                "Action": [
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:StopQueryExecution",
                    "athena:GetWorkGroup",
                    "athena:ListWorkGroups"
                ],
                "Resource": "*"
            },
            {
                "Sid": "LakeFormationAccess",
                "Effect": "Allow",
                "Action": [
                    "lakeformation:GetDataAccess",
                    "lakeformation:GetResourceLFTags",
                    "lakeformation:ListLFTags",
                    "lakeformation:GetLFTag",
                    "lakeformation:SearchTablesByLFTags",
                    "lakeformation:SearchDatabasesByLFTags",
                    "lakeformation:GetWorkUnits",
                    "lakeformation:GetWorkUnitResults",
                    "lakeformation:StartQueryPlanning",
                    "lakeformation:GetQueryState",
                    "lakeformation:GetQueryStatistics"
                ],
                "Resource": "*"
            }
        ]
    }


def grant_lake_formation_permissions(role_arn: str, database_name: str = "fraud_detection", region: str = "us-east-1") -> bool:
    """
    Grant Lake Formation permissions to the role for the specified database.
    
    Args:
        role_arn: IAM role ARN
        database_name: Name of the database to grant access to
        region: AWS region
        
    Returns:
        True if successful, False otherwise
    """
    lf_client = boto3.client('lakeformation', region_name=region)
    
    logger.info(f"Granting Lake Formation permissions for database: {database_name}")
    
    try:
        # Grant database permissions
        logger.info("  Granting database permissions...")
        try:
            lf_client.grant_permissions(
                Principal={'DataLakePrincipalIdentifier': role_arn},
                Resource={'Database': {'Name': database_name}},
                Permissions=['DESCRIBE', 'ALTER', 'CREATE_TABLE', 'DROP']
            )
            logger.info("    ✓ Database permissions granted")
        except ClientError as e:
            if 'AlreadyExists' in str(e) or 'already exists' in str(e).lower():
                logger.info("    ○ Database permissions already exist")
            else:
                logger.warning(f"    ✗ Failed to grant database permissions: {e}")
        
        # Grant table permissions (all tables in database) - try wildcard first
        logger.info("  Granting table permissions (wildcard)...")
        wildcard_success = False
        try:
            lf_client.grant_permissions(
                Principal={'DataLakePrincipalIdentifier': role_arn},
                Resource={'Table': {'DatabaseName': database_name, 'TableWildcard': {}}},
                Permissions=['SELECT', 'DESCRIBE', 'ALTER', 'DELETE', 'INSERT']
            )
            logger.info("    ✓ Table wildcard permissions granted")
            wildcard_success = True
        except ClientError as e:
            if 'AlreadyExists' in str(e) or 'already exists' in str(e).lower():
                logger.info("    ○ Table wildcard permissions already exist")
                wildcard_success = True
            else:
                logger.warning(f"    ○ Wildcard permissions failed (will try specific tables): {e}")
        
        # If wildcard failed, grant permissions on specific tables
        if not wildcard_success:
            logger.info("  Granting permissions on specific tables...")
            tables = ['training_data', 'ground_truth', 'inference_responses', 'ground_truth_updates']
            for table in tables:
                try:
                    lf_client.grant_permissions(
                        Principal={'DataLakePrincipalIdentifier': role_arn},
                        Resource={'Table': {'DatabaseName': database_name, 'Name': table}},
                        Permissions=['SELECT', 'DESCRIBE', 'ALTER', 'DELETE', 'INSERT']
                    )
                    logger.info(f"    ✓ Permissions granted for table: {table}")
                except ClientError as e:
                    if 'AlreadyExists' in str(e) or 'already exists' in str(e).lower():
                        logger.info(f"    ○ Permissions already exist for table: {table}")
                    elif 'EntityNotFound' in str(e):
                        logger.info(f"    ○ Table not found (may not exist yet): {table}")
                    else:
                        logger.warning(f"    ✗ Failed to grant permissions for {table}: {e}")
        
        return True
        
    except ClientError as e:
        logger.error(f"Failed to grant Lake Formation permissions: {e}")
        return False


def get_role_name_from_arn(role_arn: str) -> str:
    """Extract role name from ARN."""
    # ARN format: arn:aws:iam::123456789:role/path/RoleName
    # or: arn:aws:iam::123456789:role/RoleName
    parts = role_arn.split('/')
    return parts[-1]


def update_role(
    role_name: str,
    region: str = "us-east-1"
) -> dict:
    """
    Update an existing IAM role with required policies.
    
    Args:
        role_name: Name of the IAM role
        region: AWS region
        
    Returns:
        Dictionary with role ARN and status
    """
    iam_client = boto3.client('iam')
    sts_client = boto3.client('sts')
    
    account_id = sts_client.get_caller_identity()['Account']
    
    logger.info("=" * 60)
    logger.info(f"Updating existing role: {role_name}")
    logger.info(f"AWS Account: {account_id}")
    logger.info("=" * 60)
    
    try:
        # Verify role exists
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response['Role']['Arn']
        logger.info(f"Found role: {role_arn}")
        
        # Get currently attached policies
        attached_policies = set()
        paginator = iam_client.get_paginator('list_attached_role_policies')
        for page in paginator.paginate(RoleName=role_name):
            for policy in page['AttachedPolicies']:
                attached_policies.add(policy['PolicyArn'])
        
        logger.info(f"Currently attached policies: {len(attached_policies)}")
        
        # Attach managed policies (skip if already attached)
        logger.info("Attaching managed policies...")
        policies_attached = 0
        policies_skipped = 0
        
        for policy_arn in MANAGED_POLICIES:
            policy_name = policy_arn.split('/')[-1]
            if policy_arn in attached_policies:
                logger.info(f"  ○ Already attached: {policy_name}")
                policies_skipped += 1
            else:
                try:
                    iam_client.attach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy_arn
                    )
                    logger.info(f"  ✓ Attached: {policy_name}")
                    policies_attached += 1
                except ClientError as e:
                    logger.warning(f"  ✗ Failed to attach {policy_name}: {e}")
        
        # Update custom inline policy
        logger.info("Updating custom inline policy...")
        custom_policy = get_custom_policy(account_id, region)
        
        try:
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName='SageMakerMLPipelineCustomPolicy',
                PolicyDocument=json.dumps(custom_policy)
            )
            logger.info("  ✓ Custom policy updated: SageMakerMLPipelineCustomPolicy")
        except ClientError as e:
            logger.error(f"  ✗ Failed to update custom policy: {e}")
        
        # Try to update trust policy (may fail if role is service-linked)
        logger.info("Updating trust policy...")
        try:
            iam_client.update_assume_role_policy(
                RoleName=role_name,
                PolicyDocument=json.dumps(TRUST_POLICY)
            )
            logger.info("  ✓ Trust policy updated (SageMaker + Lambda)")
        except ClientError as e:
            logger.warning(f"  ○ Could not update trust policy: {e}")
            logger.warning("    (This is normal for service-linked roles)")
        
        # Grant Lake Formation permissions
        logger.info("Granting Lake Formation permissions...")
        grant_lake_formation_permissions(role_arn, "fraud_detection", region)
        
        logger.info("=" * 60)
        logger.info("✓ Role updated successfully!")
        logger.info(f"  Role ARN: {role_arn}")
        logger.info(f"  Policies attached: {policies_attached}")
        logger.info(f"  Policies already present: {policies_skipped}")
        logger.info("=" * 60)
        
        return {
            'role_arn': role_arn,
            'role_name': role_name,
            'status': 'updated',
            'policies_attached': policies_attached,
            'policies_skipped': policies_skipped,
            'account_id': account_id
        }
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            logger.error(f"Role {role_name} does not exist")
            return {
                'role_name': role_name,
                'status': 'not_found',
                'error': 'Role does not exist'
            }
        raise


def create_role(
    role_name: str,
    region: str = "us-east-1",
    description: str = "SageMaker execution role with Lambda support for ML pipelines"
) -> dict:
    """
    Create IAM role with required policies.
    
    Args:
        role_name: Name of the IAM role
        region: AWS region
        description: Role description
        
    Returns:
        Dictionary with role ARN and status
    """
    iam_client = boto3.client('iam')
    sts_client = boto3.client('sts')
    
    account_id = sts_client.get_caller_identity()['Account']
    
    logger.info("=" * 60)
    logger.info(f"Creating new role: {role_name}")
    logger.info(f"AWS Account: {account_id}")
    logger.info("=" * 60)
    
    try:
        # Check if role already exists
        try:
            existing_role = iam_client.get_role(RoleName=role_name)
            logger.warning(f"Role {role_name} already exists, updating instead...")
            return update_role(role_name, region)
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchEntity':
                raise
        
        # Create the role
        logger.info("Creating IAM role...")
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description=description,
            Tags=[
                {'Key': 'Purpose', 'Value': 'SageMaker-ML-Pipeline'},
                {'Key': 'ManagedBy', 'Value': 'create_or_update_sagemaker_role.py'},
            ]
        )
        role_arn = response['Role']['Arn']
        logger.info(f"✓ Role created: {role_arn}")
        
        # Wait for role to be available
        logger.info("Waiting for role to be available...")
        time.sleep(5)
        
        # Attach managed policies
        logger.info("Attaching managed policies...")
        for policy_arn in MANAGED_POLICIES:
            try:
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
                policy_name = policy_arn.split('/')[-1]
                logger.info(f"  ✓ Attached: {policy_name}")
            except ClientError as e:
                logger.warning(f"  ✗ Failed to attach {policy_arn}: {e}")
        
        # Create and attach custom inline policy
        logger.info("Creating custom inline policy...")
        custom_policy = get_custom_policy(account_id, region)
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName='SageMakerMLPipelineCustomPolicy',
            PolicyDocument=json.dumps(custom_policy)
        )
        logger.info("  ✓ Custom policy attached")
        
        # Grant Lake Formation permissions
        logger.info("Granting Lake Formation permissions...")
        grant_lake_formation_permissions(role_arn, "fraud_detection", region)
        
        logger.info("=" * 60)
        logger.info("✓ Role created successfully!")
        logger.info(f"  Role ARN: {role_arn}")
        logger.info("=" * 60)
        
        return {
            'role_arn': role_arn,
            'role_name': role_name,
            'status': 'created',
            'account_id': account_id
        }
        
    except ClientError as e:
        logger.error(f"Failed to create role: {e}")
        raise


def update_env_file(role_arn: str, env_path: Path = None) -> None:
    """
    Update .env file with the role ARN.
    
    Args:
        role_arn: The role ARN to add
        env_path: Path to .env file
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
        return
    
    content = env_path.read_text()
    
    if 'SAGEMAKER_EXEC_ROLE=' in content:
        lines = content.split('\n')
        updated_lines = []
        for line in lines:
            if line.startswith('SAGEMAKER_EXEC_ROLE='):
                updated_lines.append(f'SAGEMAKER_EXEC_ROLE="{role_arn}"')
                logger.info(f"Updated SAGEMAKER_EXEC_ROLE in {env_path}")
            else:
                updated_lines.append(line)
        content = '\n'.join(updated_lines)
    else:
        content += f'\n\n# SageMaker Execution Role\nSAGEMAKER_EXEC_ROLE="{role_arn}"\n'
        logger.info(f"Added SAGEMAKER_EXEC_ROLE to {env_path}")
    
    env_path.write_text(content)


def delete_role(role_name: str) -> dict:
    """
    Delete IAM role and all attached policies.
    
    Args:
        role_name: Name of the IAM role to delete
        
    Returns:
        Dictionary with deletion status
    """
    iam_client = boto3.client('iam')
    
    logger.info(f"Deleting role: {role_name}")
    
    try:
        # Detach managed policies
        logger.info("Detaching managed policies...")
        paginator = iam_client.get_paginator('list_attached_role_policies')
        for page in paginator.paginate(RoleName=role_name):
            for policy in page['AttachedPolicies']:
                iam_client.detach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy['PolicyArn']
                )
                logger.info(f"  ✓ Detached: {policy['PolicyName']}")
        
        # Delete inline policies
        logger.info("Deleting inline policies...")
        paginator = iam_client.get_paginator('list_role_policies')
        for page in paginator.paginate(RoleName=role_name):
            for policy_name in page['PolicyNames']:
                iam_client.delete_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name
                )
                logger.info(f"  ✓ Deleted: {policy_name}")
        
        # Delete the role
        iam_client.delete_role(RoleName=role_name)
        logger.info(f"✓ Role {role_name} deleted successfully")
        
        return {
            'role_name': role_name,
            'status': 'deleted'
        }
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            logger.warning(f"Role {role_name} does not exist")
            return {
                'role_name': role_name,
                'status': 'not_found'
            }
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update SageMaker Execution Role with required policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update existing role from .env (SAGEMAKER_EXEC_ROLE)
  python scripts/create_or_update_sagemaker_role.py

  # Update a specific role by ARN
  python scripts/create_or_update_sagemaker_role.py --role-arn arn:aws:iam::123456789:role/MyRole

  # Update a specific role by name
  python scripts/create_or_update_sagemaker_role.py --role-name MyRoleName

  # Create a new role (requires --create flag)
  python scripts/create_or_update_sagemaker_role.py --create --role-name fraud-detection-sagemaker-role

  # Delete an existing role
  python scripts/create_or_update_sagemaker_role.py --role-name my-role --delete
        """
    )

    parser.add_argument(
        '--role-name',
        help='Name of the IAM role to update'
    )
    parser.add_argument(
        '--role-arn',
        help='ARN of existing role to update (overrides .env)'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create a new role (requires --role-name). Default behavior is UPDATE only.'
    )
    parser.add_argument(
        '--update-env',
        action='store_true',
        help='Update .env file with the role ARN'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Delete the role instead of updating it'
    )

    args = parser.parse_args()

    try:
        # Determine which role to work with
        role_arn = args.role_arn or os.getenv('SAGEMAKER_EXEC_ROLE')
        role_name = args.role_name

        if args.delete:
            # DELETE mode
            if not role_name:
                if role_arn:
                    role_name = get_role_name_from_arn(role_arn)
                else:
                    logger.error("Please specify --role-name or --role-arn for deletion")
                    return 1
            result = delete_role(role_name)

        elif args.create:
            # CREATE mode (explicit flag required)
            if not role_name:
                logger.error("--role-name is required when using --create")
                return 1
            logger.info(f"CREATE mode: Creating new role {role_name}")
            result = create_role(role_name, args.region)

            # Auto-update .env when creating a new role
            if result['status'] == 'created':
                update_env_file(result['role_arn'])
                logger.info("✓ .env file updated with new role ARN")

        else:
            # UPDATE mode (default)
            if role_arn:
                # Update existing role from ARN
                role_name = get_role_name_from_arn(role_arn)
                logger.info(f"UPDATE mode: Found role in environment/args: {role_arn}")
                result = update_role(role_name, args.region)
            elif role_name:
                # Update existing role by name
                logger.info(f"UPDATE mode: Updating role by name: {role_name}")
                result = update_role(role_name, args.region)
            else:
                # No role specified - error
                logger.error("No role specified!")
                logger.error("Please either:")
                logger.error("  1. Set SAGEMAKER_EXEC_ROLE in .env file")
                logger.error("  2. Use --role-arn or --role-name argument")
                logger.error("  3. Use --create --role-name to create a new role")
                return 1

            # Handle not found case
            if result.get('status') == 'not_found':
                logger.error(f"Role '{role_name}' does not exist!")
                logger.error("To create it, use: --create --role-name {role_name}")
                return 1

        print(json.dumps(result, indent=2))
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
