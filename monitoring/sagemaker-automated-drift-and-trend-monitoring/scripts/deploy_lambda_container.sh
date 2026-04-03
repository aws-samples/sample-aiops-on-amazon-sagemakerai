#!/bin/bash
set -e

#############################################################################
# Deploy Drift Monitoring Lambda as Container Image
#############################################################################
# This script:
# 1. Creates SNS topic for alerts
# 2. Creates IAM role for Lambda
# 3. Builds and deploys Lambda as container image to ECR
# 4. Creates EventBridge schedule rule
# 5. Tests the Lambda function
#############################################################################

REGION="${AWS_REGION:-us-east-1}"
ALERT_EMAIL="${1:-}"  # Pass email as first argument
DATA_DRIFT_THRESHOLD="${2:-0.2}"
MODEL_DRIFT_THRESHOLD="${3:-0.05}"

# Validate email
if [ -z "$ALERT_EMAIL" ]; then
    echo "Usage: $0 <email> [data_drift_threshold] [model_drift_threshold]"
    echo "Example: $0 your-email@example.com 0.2 0.05"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Deploying Drift Monitoring Infrastructure                         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Region: $REGION"
echo "  Alert Email: $ALERT_EMAIL"
echo "  Data Drift Threshold: $DATA_DRIFT_THRESHOLD"
echo "  Model Drift Threshold: $MODEL_DRIFT_THRESHOLD"
echo ""

# Load configuration from .env if available
if [ -f ../.env ]; then
    set -a
    source ../.env
    set +a
fi

# Get AWS account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Configuration with environment variable overrides
ATHENA_DB="${ATHENA_DATABASE:-fraud_detection}"
ATHENA_OUTPUT="s3://${DATA_S3_BUCKET:-fraud-detection-data-lake-skoppar-${ACCOUNT_ID}}/athena-query-results/"
SNS_TOPIC="${SNS_TOPIC_NAME:-fraud-detection-drift-alerts}"
LAMBDA_NAME="${DRIFT_LAMBDA_NAME:-fraud-detection-drift-monitor}"
ROLE_NAME=$(echo "${LAMBDA_EXEC_ROLE:-fraud-detection-drift-monitor-role}" | awk -F'/' '{print $NF}')
RULE_NAME="${EVENTBRIDGE_RULE_NAME:-fraud-detection-drift-check}"
SCHEDULE="${EVENTBRIDGE_SCHEDULE:-cron(0 2 * * ? *)}"  # Daily at 2 AM UTC
REPO_NAME="$LAMBDA_NAME"  # ECR repo name matches Lambda name

# Step 1: SNS Topic
echo "[1/7] Creating SNS topic..."
TOPIC_ARN=$(aws sns create-topic --name $SNS_TOPIC --region $REGION --query 'TopicArn' --output text 2>/dev/null || \
            aws sns list-topics --region $REGION --query "Topics[?contains(TopicArn, '$SNS_TOPIC')].TopicArn" --output text)
echo "  ✓ Topic: $TOPIC_ARN"

if [ -n "$ALERT_EMAIL" ]; then
    aws sns subscribe --topic-arn $TOPIC_ARN --protocol email --notification-endpoint $ALERT_EMAIL --region $REGION 2>/dev/null || true
    echo "  ✓ Check $ALERT_EMAIL for confirmation link"
fi

# Step 2: IAM Role
echo ""
echo "[2/7] Creating IAM role..."
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF
)

aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document "$TRUST_POLICY" \
    --description "Drift monitoring Lambda role" \
    --region $REGION 2>/dev/null || echo "  (role already exists)"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Attach managed policies
for policy in \
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
    "arn:aws:iam::aws:policy/AmazonAthenaFullAccess" \
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"; do
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn $policy --region $REGION 2>/dev/null || true
done

# Add SNS publish permission
SNS_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["sns:Publish"],
    "Resource": "$TOPIC_ARN"
  }]
}
EOF
)

aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name SNSPublishPolicy \
    --policy-document "$SNS_POLICY" \
    --region $REGION 2>/dev/null || true

echo "  ✓ Role: $ROLE_ARN"
echo "  ✓ Waiting 10s for role propagation..."
sleep 10

# Step 3: Create ECR repository if needed
echo ""
echo "[3/7] Setting up ECR repository..."
aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION 2>/dev/null || \
aws ecr create-repository \
    --repository-name $REPO_NAME \
    --region $REGION \
    --image-scanning-configuration scanOnPush=false > /dev/null
echo "  ✓ Repository: $REPO_NAME"

# Step 4: Build and push Docker image
echo ""
echo "[4/7] Building Docker image..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker build --platform linux/amd64 -t $REPO_NAME:latest -f src/drift_monitoring/Dockerfile.lambda . -q
docker tag $REPO_NAME:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest
echo "  ✓ Image built"

echo "  Pushing to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest | tail -5
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest"
echo "  ✓ Image pushed: $IMAGE_URI"

# Step 5: Create/Update Lambda function
echo ""
echo "[5/7] Deploying Lambda function..."

# Get MLflow tracking URI if available
MLFLOW_URI=$(aws sagemaker list-mlflow-tracking-servers --region $REGION --query 'TrackingServerSummaries[0].TrackingServerArn' --output text 2>/dev/null || echo "")
SQS_QUEUE_URL=$(aws sqs get-queue-url --queue-name fraud-monitoring-results --region $REGION --query 'QueueUrl' --output text 2>/dev/null || echo "")

ENV_VARS=$(cat <<EOF
{
  "ATHENA_DATABASE": "$ATHENA_DB",
  "ATHENA_OUTPUT_S3": "$ATHENA_OUTPUT",
  "SNS_TOPIC_ARN": "$TOPIC_ARN",
  "MLFLOW_TRACKING_URI": "$MLFLOW_URI",
  "BASELINE_ROC_AUC": "0.92",
  "DATA_DRIFT_THRESHOLD": "$DATA_DRIFT_THRESHOLD",
  "KS_PVALUE_THRESHOLD": "0.05",
  "MODEL_DRIFT_THRESHOLD": "$MODEL_DRIFT_THRESHOLD",
  "MONITORING_SQS_QUEUE_URL": "$SQS_QUEUE_URL"
}
EOF
)

# Check if function exists
FUNCTION_EXISTS=$(aws lambda get-function --function-name $LAMBDA_NAME --region $REGION 2>/dev/null && echo "true" || echo "false")

if [ "$FUNCTION_EXISTS" = "true" ]; then
    CURRENT_PKG_TYPE=$(aws lambda get-function-configuration --function-name $LAMBDA_NAME --region $REGION --query 'PackageType' --output text)

    if [ "$CURRENT_PKG_TYPE" = "Zip" ]; then
        echo "  Function exists with PackageType=Zip, recreating as Image..."
        # Remove permission first
        aws lambda remove-permission --function-name $LAMBDA_NAME --statement-id AllowEventBridgeInvoke --region $REGION 2>/dev/null || true
        # Delete function
        aws lambda delete-function --function-name $LAMBDA_NAME --region $REGION
        sleep 3
        FUNCTION_EXISTS="false"
    else
        echo "  Updating existing function..."
        aws lambda update-function-code --function-name $LAMBDA_NAME --image-uri $IMAGE_URI --region $REGION > /dev/null
        aws lambda wait function-updated-v2 --function-name $LAMBDA_NAME --region $REGION
        aws lambda update-function-configuration \
            --function-name $LAMBDA_NAME \
            --timeout 300 \
            --memory-size 512 \
            --environment "Variables=$ENV_VARS" \
            --region $REGION > /dev/null
        aws lambda wait function-updated-v2 --function-name $LAMBDA_NAME --region $REGION
    fi
fi

if [ "$FUNCTION_EXISTS" = "false" ]; then
    echo "  Creating new function..."
    aws lambda create-function \
        --function-name $LAMBDA_NAME \
        --package-type Image \
        --code ImageUri=$IMAGE_URI \
        --role $ROLE_ARN \
        --timeout 300 \
        --memory-size 512 \
        --description "Automated drift detection with Evidently + MLflow" \
        --environment "Variables=$ENV_VARS" \
        --region $REGION > /dev/null
    aws lambda wait function-active-v2 --function-name $LAMBDA_NAME --region $REGION
fi

FUNCTION_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${LAMBDA_NAME}"
echo "  ✓ Lambda: $FUNCTION_ARN"

# Step 6: EventBridge Rule
echo ""
echo "[6/7] Creating EventBridge schedule..."
aws events put-rule \
    --name $RULE_NAME \
    --schedule-expression "$SCHEDULE" \
    --state ENABLED \
    --description "Trigger drift monitoring Lambda" \
    --region $REGION > /dev/null

aws events put-targets \
    --rule $RULE_NAME \
    --targets "Id=1,Arn=$FUNCTION_ARN" \
    --region $REGION > /dev/null

aws lambda add-permission \
    --function-name $LAMBDA_NAME \
    --statement-id AllowEventBridgeInvoke \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
    --region $REGION 2>/dev/null || true

echo "  ✓ Schedule: $SCHEDULE"

# Step 7: Test Lambda
echo ""
echo "[7/7] Testing Lambda function..."
aws lambda invoke \
    --function-name $LAMBDA_NAME \
    --region $REGION \
    /tmp/lambda_test_output.json > /tmp/lambda_test_response.json 2>&1

STATUS_CODE=$(jq -r '.StatusCode' /tmp/lambda_test_response.json)
if [ "$STATUS_CODE" = "200" ]; then
    RESULT=$(jq -r '.statusCode' /tmp/lambda_test_output.json 2>/dev/null || echo "error")
    if [ "$RESULT" = "200" ]; then
        echo "  ✓ Lambda test successful!"
        jq '.' /tmp/lambda_test_output.json | head -20
    else
        echo "  ⚠ Lambda executed but returned error:"
        jq '.' /tmp/lambda_test_output.json
    fi
else
    echo "  ❌ Lambda test failed:"
    cat /tmp/lambda_test_response.json
fi

# Save configuration
echo ""
echo "Saving deployment configuration..."
CONFIG_FILE="src/config/drift_monitoring_config.json"
cat > $CONFIG_FILE <<EOF
{
  "sns_topic_arn": "$TOPIC_ARN",
  "lambda_function_arn": "$FUNCTION_ARN",
  "eventbridge_rule_arn": "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}",
  "ecr_image_uri": "$IMAGE_URI",
  "schedule": "$SCHEDULE",
  "data_drift_threshold": "$DATA_DRIFT_THRESHOLD",
  "model_drift_threshold": "$MODEL_DRIFT_THRESHOLD",
  "email": "$ALERT_EMAIL",
  "region": "$REGION"
}
EOF
echo "  ✓ Config saved: $CONFIG_FILE"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ DEPLOYMENT COMPLETE                                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  SNS Topic:     $TOPIC_ARN"
echo "  Lambda:        $FUNCTION_ARN"
echo "  EventBridge:   $RULE_NAME ($SCHEDULE)"
echo "  Docker Image:  $IMAGE_URI"
echo ""
echo "Next steps:"
echo "  1. Check $ALERT_EMAIL for SNS subscription confirmation"
echo "  2. Monitor: aws logs tail /aws/lambda/$LAMBDA_NAME --follow"
echo "  3. Manual test: aws lambda invoke --function-name $LAMBDA_NAME output.json"
echo ""
