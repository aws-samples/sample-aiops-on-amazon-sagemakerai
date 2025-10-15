import boto3
import json
from typing import Optional

def add_sagemaker_mlflow_s3_permissions(role_arn: str):
    """Add SageMaker MLflow and S3 permissions to IAM role"""
    iam = boto3.client('iam')
    role_name = role_arn.split('/')[-1]
    
    # SageMaker MLflow policy
    sagemaker_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sagemaker-mlflow:*",
                "Resource": "*"
            }
        ]
    }
    
    # S3 full access policy
    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:*",
                "Resource": "*"
            }
        ]
    }
    
    # Create and attach policies
    try:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName='SageMakerMLflowAccessAgent',
            PolicyDocument=json.dumps(sagemaker_policy)
        )
        print(f"✓ Added SageMaker MLflow permissions to {role_name}")
        
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName='S3FullAccessAgent',
            PolicyDocument=json.dumps(s3_policy)
        )
        print(f"✓ Added S3 full access permissions to {role_name}")
        
    except Exception as e:
        print(f"Error adding permissions: {e}")