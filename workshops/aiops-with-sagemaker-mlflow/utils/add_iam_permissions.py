import boto3
import json
from typing import Optional

def add_sagemaker_mlflow_s3_permissions(role_arn: str):
    """Add SageMaker MLflow App, S3, and STS permissions to IAM role"""
    iam = boto3.client('iam')
    role_name = role_arn.split('/')[-1]
    
    # SageMaker MLflow App policy
    sagemaker_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sagemaker-mlflow:*",
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sagemaker:DescribeMlflowApp",
                    "sagemaker:CallMlflowAppApi",
                    "sagemaker:CreatePresignedMlflowTrackingServerUrl",
                    "sagemaker:CreateMlflowTrackingServer",
                    "sagemaker:ListMlflowTrackingServers",
                    "sagemaker:DescribeMlflowTrackingServer"
                ],
                "Resource": "*"
            },
            {
                "Sid": "AllowSelfAssumeRole",
                "Effect": "Allow",
                "Action": "sts:AssumeRole",
                "Resource": role_arn
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
    
    # Update trust policy to allow self-assumption and SageMaker service
    try:
        current_trust = iam.get_role(RoleName=role_name)['Role']['AssumeRolePolicyDocument']
        
        # Add self-assume statement
        self_assume = {
            "Effect": "Allow",
            "Principal": {"AWS": role_arn},
            "Action": "sts:AssumeRole"
        }
        
        # Add SageMaker service principal
        sagemaker_assume = {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
        
        statements_to_add = []
        existing_json = json.dumps(current_trust)
        
        if role_arn not in existing_json:
            statements_to_add.append(self_assume)
        if "sagemaker.amazonaws.com" not in existing_json:
            statements_to_add.append(sagemaker_assume)
        
        if statements_to_add:
            current_trust['Statement'].extend(statements_to_add)
            iam.update_assume_role_policy(
                RoleName=role_name,
                PolicyDocument=json.dumps(current_trust)
            )
            print(f"✓ Updated trust policy on {role_name}")
        else:
            print(f"✓ Trust policy already has required principals for {role_name}")
    except Exception as e:
        print(f"Warning: Could not update trust policy: {e}")
    
    # Create and attach policies
    try:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName='SageMakerMLflowAccessAgent',
            PolicyDocument=json.dumps(sagemaker_policy)
        )
        print(f"✓ Added SageMaker MLflow App permissions to {role_name}")
        
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName='S3FullAccessAgent',
            PolicyDocument=json.dumps(s3_policy)
        )
        print(f"✓ Added S3 full access permissions to {role_name}")
        
    except Exception as e:
        print(f"Error adding permissions: {e}")
