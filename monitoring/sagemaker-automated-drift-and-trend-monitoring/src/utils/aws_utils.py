"""AWS utility functions for SageMaker operations."""

import os
import boto3
import sagemaker
from botocore.exceptions import ClientError, NoCredentialsError
from sagemaker import Session


def get_sagemaker_session() -> Session:
    """
    Get a SageMaker session.

    Returns:
        SageMaker Session object
    """
    # Ensure AWS region is set (default to us-east-1 if not set)
    if not os.environ.get('AWS_DEFAULT_REGION') and not os.environ.get('AWS_REGION'):
        boto_session = boto3.Session()
        if boto_session.region_name is None:
            print("No AWS region configured, using us-east-1 as default")
            boto_session = boto3.Session(region_name='us-east-1')
        return sagemaker.Session(boto_session=boto_session)

    return sagemaker.Session()


def get_execution_role(sagemaker_session: Session = None) -> str:
    """
    Get the SageMaker execution role.

    Args:
        sagemaker_session: Optional SageMaker session to use

    Returns:
        IAM role ARN for SageMaker execution

    Note:
        Priority order:
        1. SAGEMAKER_EXEC_ROLE environment variable (from .env)
        2. SageMaker execution environment
        3. Derived from AWS caller identity
    """
    # First, try to get role from environment variable
    role_from_env = os.environ.get('SAGEMAKER_EXEC_ROLE')
    if role_from_env:
        print(f"Using SageMaker execution role from environment: {role_from_env}")
        return role_from_env

    # Try to get role from SageMaker environment or caller identity
    try:
        role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
        print(f"Using SageMaker execution role: {role}")
        return role
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')

        if error_code == 'ExpiredToken':
            print("\n" + "=" * 80)
            print("ERROR: AWS SSO token has expired")
            print("=" * 80)
            print("\nPlease refresh your SSO session:")
            print("  aws sso login --profile default")
            print("\nOr if using a specific profile:")
            print("  aws sso login --profile <your-profile-name>")
            print("\nAfter logging in, run the deployment command again.")
            print("=" * 80)
        else:
            print(f"Warning: Could not get SageMaker execution role: {e}")
            print("You may need to specify a role ARN manually")

        raise
    except NoCredentialsError:
        print("\n" + "=" * 80)
        print("ERROR: No AWS credentials found")
        print("=" * 80)
        print("\nPlease configure AWS credentials:")
        print("  1. For SSO: aws sso login --profile default")
        print("  2. For access keys: aws configure")
        print("=" * 80)
        raise
    except Exception as e:
        print(f"Warning: Could not get SageMaker execution role: {e}")
        print("You may need to specify a role ARN manually")
        raise
