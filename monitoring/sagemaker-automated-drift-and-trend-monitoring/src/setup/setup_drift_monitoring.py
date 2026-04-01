#!/usr/bin/env python3
"""
Setup automated drift monitoring with EventBridge, Lambda, and SNS.

This script creates:
1. SNS topic for drift alerts
2. Email subscription to SNS topic
3. Lambda function for drift detection
4. IAM role for Lambda with necessary permissions
5. EventBridge rule to trigger Lambda on schedule
"""

import boto3
import json
import os
import zipfile
import time
from pathlib import Path

# AWS clients
iam = boto3.client('iam')
sns = boto3.client('sns')
lambda_client = boto3.client('lambda')
events = boto3.client('events')
sts = boto3.client('sts')

# Configuration
from src.config.config import AWS_DEFAULT_REGION
AWS_ACCOUNT_ID = sts.get_caller_identity()['Account']

SNS_TOPIC_NAME = 'fraud-detection-drift-alerts'
LAMBDA_FUNCTION_NAME = 'fraud-detection-drift-monitor'
LAMBDA_ROLE_NAME = 'fraud-detection-drift-monitor-role'
EVENTBRIDGE_RULE_NAME = 'fraud-detection-drift-check'

# Schedule: Run daily at 2 AM UTC
SCHEDULE_EXPRESSION = 'cron(0 2 * * ? *)'  # Daily at 2 AM UTC

# Environment variables for Lambda
ATHENA_DATABASE = os.getenv('ATHENA_DATABASE', 'fraud_detection')
ATHENA_OUTPUT_S3 = os.getenv('ATHENA_OUTPUT_S3', f's3://fraud-detection-data-lake-skoppar-{AWS_ACCOUNT_ID}/athena-query-results/')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', '')
BASELINE_ROC_AUC = os.getenv('BASELINE_ROC_AUC', '0.92')

# Thresholds
DATA_DRIFT_THRESHOLD = os.getenv('DATA_DRIFT_THRESHOLD', '0.2')
MODEL_DRIFT_THRESHOLD = os.getenv('MODEL_DRIFT_THRESHOLD', '0.05')


def create_sns_topic(email_address):
    """Create SNS topic and subscribe email."""
    print(f"\n📧 Creating SNS topic: {SNS_TOPIC_NAME}")

    try:
        response = sns.create_topic(Name=SNS_TOPIC_NAME)
        topic_arn = response['TopicArn']
        print(f"✓ SNS topic created: {topic_arn}")
    except sns.exceptions.TopicLimitExceeded:
        # Topic already exists
        topics = sns.list_topics()
        topic_arn = next(
            (t['TopicArn'] for t in topics['Topics'] if SNS_TOPIC_NAME in t['TopicArn']),
            None
        )
        print(f"✓ SNS topic already exists: {topic_arn}")

    # Subscribe email
    if email_address:
        print(f"\n📬 Subscribing email: {email_address}")
        try:
            sns.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email_address
            )
            print(f"✓ Email subscription created (check inbox for confirmation)")
        except Exception as e:
            print(f"⚠️ Email subscription failed: {e}")

    return topic_arn


def create_lambda_role(topic_arn):
    """Create IAM role for Lambda with necessary permissions."""
    print(f"\n🔐 Creating IAM role: {LAMBDA_ROLE_NAME}")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        role = iam.create_role(
            RoleName=LAMBDA_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Role for fraud detection drift monitoring Lambda'
        )
        role_arn = role['Role']['Arn']
        print(f"✓ IAM role created: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role = iam.get_role(RoleName=LAMBDA_ROLE_NAME)
        role_arn = role['Role']['Arn']
        print(f"✓ IAM role already exists: {role_arn}")

    # Attach managed policies
    managed_policies = [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/AmazonAthenaFullAccess',
        'arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess',
    ]

    for policy_arn in managed_policies:
        try:
            iam.attach_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyArn=policy_arn)
            print(f"  ✓ Attached: {policy_arn.split('/')[-1]}")
        except iam.exceptions.LimitExceededException:
            print(f"  ⚠️ Policy already attached: {policy_arn.split('/')[-1]}")

    # Create inline policy for SNS publish
    sns_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["sns:Publish"],
                "Resource": topic_arn
            }
        ]
    }

    try:
        iam.put_role_policy(
            RoleName=LAMBDA_ROLE_NAME,
            PolicyName='SNSPublishPolicy',
            PolicyDocument=json.dumps(sns_policy)
        )
        print(f"  ✓ Added SNS publish policy")
    except Exception as e:
        print(f"  ⚠️ SNS policy error: {e}")

    # Wait for role to propagate
    print("  ⏳ Waiting for IAM role to propagate...")
    time.sleep(10)

    return role_arn


def create_lambda_deployment_package():
    """Create Lambda deployment package with dependencies."""
    print(f"\n📦 Creating Lambda deployment package")

    project_root = Path(__file__).parent.parent.parent
    lambda_code_path = project_root / 'src' / 'scripts' / 'lambda_drift_monitor.py'

    if not lambda_code_path.exists():
        raise FileNotFoundError(f"Lambda code not found: {lambda_code_path}")

    # Create temporary directory for package
    package_dir = project_root / '.lambda_package'
    package_dir.mkdir(exist_ok=True)

    # Install dependencies
    print("  📥 Installing dependencies...")
    requirements_file = project_root / 'src' / 'scripts' / 'lambda_requirements.txt'
    if requirements_file.exists():
        os.system(f'pip install -q -t {package_dir} -r {requirements_file}')
    else:
        # Fallback to manual installation
        os.system(f'pip install -q -t {package_dir} scikit-learn numpy pandas mlflow matplotlib boto3')

    # Copy Lambda code
    import shutil
    shutil.copy(lambda_code_path, package_dir / 'lambda_drift_monitor.py')

    # Create ZIP file
    zip_path = project_root / 'lambda_drift_monitor.zip'
    print(f"  🗜️ Creating ZIP file: {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)

    # Cleanup
    shutil.rmtree(package_dir)

    print(f"✓ Lambda package created: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    return zip_path


def create_lambda_function(role_arn, topic_arn, zip_path):
    """Create or update Lambda function."""
    print(f"\n⚡ Creating Lambda function: {LAMBDA_FUNCTION_NAME}")

    with open(zip_path, 'rb') as f:
        zip_content = f.read()

    environment = {
        'Variables': {
            'ATHENA_DATABASE': ATHENA_DATABASE,
            'ATHENA_OUTPUT_S3': ATHENA_OUTPUT_S3,
            'SNS_TOPIC_ARN': topic_arn,
            'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI,
            'BASELINE_ROC_AUC': BASELINE_ROC_AUC,
            'DATA_DRIFT_THRESHOLD': DATA_DRIFT_THRESHOLD,
            'MODEL_DRIFT_THRESHOLD': MODEL_DRIFT_THRESHOLD,
        }
    }

    try:
        response = lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Runtime='python3.11',
            Role=role_arn,
            Handler='lambda_drift_monitor.lambda_handler',
            Code={'ZipFile': zip_content},
            Description='Automated drift detection and alerting',
            Timeout=300,  # 5 minutes
            MemorySize=512,
            Environment=environment
        )
        function_arn = response['FunctionArn']
        print(f"✓ Lambda function created: {function_arn}")
    except lambda_client.exceptions.ResourceConflictException:
        # Update existing function
        print("  ⚠️ Function exists, updating code...")
        lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION_NAME,
            ZipFile=zip_content
        )
        lambda_client.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Environment=environment,
            Timeout=300,
            MemorySize=512
        )
        response = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        function_arn = response['Configuration']['FunctionArn']
        print(f"✓ Lambda function updated: {function_arn}")

    return function_arn


def create_eventbridge_rule(function_arn):
    """Create EventBridge rule to trigger Lambda on schedule."""
    print(f"\n⏰ Creating EventBridge rule: {EVENTBRIDGE_RULE_NAME}")

    try:
        rule_response = events.put_rule(
            Name=EVENTBRIDGE_RULE_NAME,
            ScheduleExpression=SCHEDULE_EXPRESSION,
            State='ENABLED',
            Description='Trigger drift monitoring Lambda daily'
        )
        rule_arn = rule_response['RuleArn']
        print(f"✓ EventBridge rule created: {rule_arn}")
        print(f"  Schedule: {SCHEDULE_EXPRESSION} (Daily at 2 AM UTC)")
    except Exception as e:
        print(f"⚠️ EventBridge rule error: {e}")
        return None

    # Add Lambda as target
    try:
        events.put_targets(
            Rule=EVENTBRIDGE_RULE_NAME,
            Targets=[
                {
                    'Id': '1',
                    'Arn': function_arn
                }
            ]
        )
        print(f"✓ Lambda added as target")
    except Exception as e:
        print(f"⚠️ Target error: {e}")

    # Add permission for EventBridge to invoke Lambda
    try:
        lambda_client.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId='AllowEventBridgeInvoke',
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=rule_arn
        )
        print(f"✓ Lambda permission granted to EventBridge")
    except lambda_client.exceptions.ResourceConflictException:
        print(f"  ⚠️ Permission already exists")

    return rule_arn


def test_lambda_function():
    """Test Lambda function with manual invocation."""
    print(f"\n🧪 Testing Lambda function...")

    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType='RequestResponse',
            LogType='Tail'
        )

        payload = json.loads(response['Payload'].read())
        print(f"✓ Lambda test completed")
        print(f"  Status Code: {payload.get('statusCode')}")
        if payload.get('body'):
            body = json.loads(payload['body'])
            print(f"  Data Drift Detected: {body.get('data_drift', {}).get('detected', 'N/A')}")
            print(f"  Model Drift Detected: {body.get('model_drift', {}).get('detected', 'N/A')}")

        return True
    except Exception as e:
        print(f"❌ Lambda test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 80)
    print("AUTOMATED DRIFT MONITORING SETUP")
    print("=" * 80)

    # Get email address for alerts
    email_address = input("\n📧 Enter email address for drift alerts: ").strip()
    if not email_address:
        print("⚠️ No email provided, skipping email subscription")
        email_address = None

    try:
        # Step 1: Create SNS topic
        topic_arn = create_sns_topic(email_address)

        # Step 2: Create Lambda IAM role
        role_arn = create_lambda_role(topic_arn)

        # Step 3: Create Lambda deployment package
        zip_path = create_lambda_deployment_package()

        # Step 4: Create Lambda function
        function_arn = create_lambda_function(role_arn, topic_arn, zip_path)

        # Step 5: Create EventBridge rule
        rule_arn = create_eventbridge_rule(function_arn)

        # Step 6: Test Lambda function
        test_success = test_lambda_function()

        print("\n" + "=" * 80)
        print("✅ SETUP COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nSNS Topic ARN: {topic_arn}")
        print(f"Lambda Function ARN: {function_arn}")
        print(f"EventBridge Rule ARN: {rule_arn}")
        print(f"\nSchedule: {SCHEDULE_EXPRESSION} (Daily at 2 AM UTC)")
        print(f"Data Drift Threshold: PSI >= {DATA_DRIFT_THRESHOLD}")
        print(f"Model Drift Threshold: {float(MODEL_DRIFT_THRESHOLD) * 100}% degradation")

        if email_address:
            print(f"\n⚠️ IMPORTANT: Check your email ({email_address}) and confirm SNS subscription!")

        print("\n📝 Next Steps:")
        print("  1. Confirm email subscription (check inbox/spam)")
        print("  2. Test manual invocation: See 2a_inference_monitoring.ipynb")
        print("  3. Wait for scheduled run or trigger manually")
        print("  4. Monitor CloudWatch logs for drift checks")

        # Save configuration
        config = {
            'sns_topic_arn': topic_arn,
            'lambda_function_arn': function_arn,
            'eventbridge_rule_arn': rule_arn,
            'schedule': SCHEDULE_EXPRESSION,
            'data_drift_threshold': DATA_DRIFT_THRESHOLD,
            'model_drift_threshold': MODEL_DRIFT_THRESHOLD,
            'email': email_address
        }

        config_path = Path(__file__).parent.parent.parent / 'drift_monitoring_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n💾 Configuration saved: {config_path}")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
