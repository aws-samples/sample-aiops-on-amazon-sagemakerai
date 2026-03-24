#!/bin/bash
set -e

#############################################################################
# Delete Drift Monitoring Infrastructure
#############################################################################
# This script safely deletes all drift monitoring resources:
# 1. EventBridge rule and targets
# 2. Lambda function
# 3. SQS queue (monitoring writer)
# 4. Lambda (monitoring writer)
# 5. SNS topic and subscriptions
# 6. IAM roles and policies
# 7. ECR repository (optional)
#############################################################################

REGION="${AWS_REGION:-us-east-1}"
DELETE_ECR="${1:-no}"  # Pass 'yes' to delete ECR images

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Deleting Drift Monitoring Infrastructure                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Region: $REGION"
echo "  Delete ECR images: $DELETE_ECR"
echo ""

# Load configuration from .env if available
if [ -f ../.env ]; then
    echo "Loading configuration from .env..."
    set -a
    source ../.env
    set +a
    echo "  ✓ Environment variables loaded"
    echo ""
fi

# Read deployment config if available
CONFIG_FILE="src/config/drift_monitoring_config.json"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading deployment configuration from $CONFIG_FILE..."
    SNS_TOPIC_ARN=$(jq -r '.sns_topic_arn' $CONFIG_FILE)
    LAMBDA_FUNCTION_ARN=$(jq -r '.lambda_function_arn' $CONFIG_FILE)
    EVENTBRIDGE_RULE=$(jq -r '.eventbridge_rule_arn' $CONFIG_FILE | sed 's/.*rule\///')
    echo "  ✓ Deployment configuration loaded"
    echo ""
else
    echo "No deployment config file found, using resource names from .env..."
    SNS_TOPIC_ARN=""
    LAMBDA_FUNCTION_ARN=""
    EVENTBRIDGE_RULE="${EVENTBRIDGE_RULE_NAME:-fraud-detection-drift-check}"
fi

# Get AWS account info
ACCOUNT_ID=$(aws sts get-caller_identity --query Account --output text)

# Resource names (from .env with defaults)
LAMBDA_NAME="${DRIFT_LAMBDA_NAME:-fraud-detection-drift-monitor}"
LAMBDA_WRITER_NAME="${MONITORING_WRITER_LAMBDA_NAME:-fraud-monitoring-results-writer}"
SNS_TOPIC="${SNS_TOPIC_NAME:-fraud-detection-drift-alerts}"
ROLE_NAME=$(echo "${LAMBDA_EXEC_ROLE:-fraud-detection-drift-monitor-role}" | awk -F'/' '{print $NF}')
ROLE_WRITER_NAME="${LAMBDA_WRITER_NAME}-role"
REPO_NAME="fraud-detection-drift-monitor"
SQS_QUEUE="fraud-monitoring-results"

# Confirmation
echo "⚠️  WARNING: This will delete the following resources:"
echo "  - EventBridge Rule: $EVENTBRIDGE_RULE"
echo "  - Lambda Function: $LAMBDA_NAME"
echo "  - Lambda Function: $LAMBDA_WRITER_NAME"
echo "  - SQS Queue: $SQS_QUEUE"
echo "  - SNS Topic: $SNS_TOPIC"
echo "  - IAM Roles: $ROLE_NAME, $ROLE_WRITER_NAME"
if [ "$DELETE_ECR" = "yes" ]; then
    echo "  - ECR Repository: $REPO_NAME (INCLUDING ALL IMAGES)"
fi
echo ""
read -p "Are you sure? (type 'yes' to confirm): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Deletion cancelled."
    exit 0
fi

echo ""
echo "Starting deletion..."
echo ""

# Step 1: EventBridge Rule
echo "[1/8] Deleting EventBridge rule..."
if aws events describe-rule --name $EVENTBRIDGE_RULE --region $REGION > /dev/null 2>&1; then
    # Remove targets first
    aws events remove-targets --rule $EVENTBRIDGE_RULE --ids 1 --region $REGION 2>/dev/null || true
    # Delete rule
    aws events delete-rule --name $EVENTBRIDGE_RULE --region $REGION
    echo "  ✓ EventBridge rule deleted"
else
    echo "  (rule not found, skipping)"
fi

# Step 2: Lambda Function (drift monitor)
echo ""
echo "[2/8] Deleting Lambda function (drift monitor)..."
if aws lambda get-function --function-name $LAMBDA_NAME --region $REGION > /dev/null 2>&1; then
    aws lambda delete-function --function-name $LAMBDA_NAME --region $REGION
    echo "  ✓ Lambda function deleted: $LAMBDA_NAME"
else
    echo "  (function not found, skipping)"
fi

# Step 3: Lambda Function (monitoring writer)
echo ""
echo "[3/8] Deleting Lambda function (monitoring writer)..."
if aws lambda get-function --function-name $LAMBDA_WRITER_NAME --region $REGION > /dev/null 2>&1; then
    aws lambda delete-function --function-name $LAMBDA_WRITER_NAME --region $REGION
    echo "  ✓ Lambda function deleted: $LAMBDA_WRITER_NAME"
else
    echo "  (function not found, skipping)"
fi

# Step 4: SQS Queue
echo ""
echo "[4/8] Deleting SQS queue..."
QUEUE_URL=$(aws sqs get-queue-url --queue-name $SQS_QUEUE --region $REGION --query 'QueueUrl' --output text 2>/dev/null || echo "")
if [ -n "$QUEUE_URL" ]; then
    aws sqs delete-queue --queue-url "$QUEUE_URL" --region $REGION
    echo "  ✓ SQS queue deleted: $SQS_QUEUE"
else
    echo "  (queue not found, skipping)"
fi

# Step 5: SNS Topic
echo ""
echo "[5/8] Deleting SNS topic..."
if [ -n "$SNS_TOPIC_ARN" ]; then
    TOPIC_ARN=$SNS_TOPIC_ARN
else
    TOPIC_ARN=$(aws sns list-topics --region $REGION --query "Topics[?contains(TopicArn, '$SNS_TOPIC')].TopicArn" --output text)
fi

if [ -n "$TOPIC_ARN" ]; then
    # Delete all subscriptions first
    SUBSCRIPTIONS=$(aws sns list-subscriptions-by-topic --topic-arn $TOPIC_ARN --region $REGION --query 'Subscriptions[].SubscriptionArn' --output text)
    for sub in $SUBSCRIPTIONS; do
        if [ "$sub" != "PendingConfirmation" ]; then
            aws sns unsubscribe --subscription-arn $sub --region $REGION 2>/dev/null || true
        fi
    done
    # Delete topic
    aws sns delete-topic --topic-arn $TOPIC_ARN --region $REGION
    echo "  ✓ SNS topic deleted: $SNS_TOPIC"
else
    echo "  (topic not found, skipping)"
fi

# Step 6: IAM Role (drift monitor)
echo ""
echo "[6/8] Deleting IAM role (drift monitor)..."
if aws iam get-role --role-name $ROLE_NAME --region $REGION > /dev/null 2>&1; then
    # Detach managed policies
    for policy in \
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
        "arn:aws:iam::aws:policy/AmazonAthenaFullAccess" \
        "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"; do
        aws iam detach-role-policy --role-name $ROLE_NAME --policy-arn $policy --region $REGION 2>/dev/null || true
    done

    # Delete inline policies
    INLINE_POLICIES=$(aws iam list-role-policies --role-name $ROLE_NAME --region $REGION --query 'PolicyNames' --output text)
    for policy in $INLINE_POLICIES; do
        aws iam delete-role-policy --role-name $ROLE_NAME --policy-name $policy --region $REGION 2>/dev/null || true
    done

    # Delete role
    aws iam delete-role --role-name $ROLE_NAME --region $REGION
    echo "  ✓ IAM role deleted: $ROLE_NAME"
else
    echo "  (role not found, skipping)"
fi

# Step 7: IAM Role (monitoring writer)
echo ""
echo "[7/8] Deleting IAM role (monitoring writer)..."
if aws iam get-role --role-name $ROLE_WRITER_NAME --region $REGION > /dev/null 2>&1; then
    # Detach managed policies
    for policy in \
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
        "arn:aws:iam::aws:policy/service-role/AWSLambdaSQSQueueExecutionRole" \
        "arn:aws:iam::aws:policy/AmazonAthenaFullAccess"; do
        aws iam detach-role-policy --role-name $ROLE_WRITER_NAME --policy-arn $policy --region $REGION 2>/dev/null || true
    done

    # Delete inline policies
    INLINE_POLICIES=$(aws iam list-role-policies --role-name $ROLE_WRITER_NAME --region $REGION --query 'PolicyNames' --output text)
    for policy in $INLINE_POLICIES; do
        aws iam delete-role-policy --role-name $ROLE_WRITER_NAME --policy-name $policy --region $REGION 2>/dev/null || true
    done

    # Delete role
    aws iam delete-role --role-name $ROLE_WRITER_NAME --region $REGION
    echo "  ✓ IAM role deleted: $ROLE_WRITER_NAME"
else
    echo "  (role not found, skipping)"
fi

# Step 8: ECR Repository (optional)
echo ""
echo "[8/8] ECR Repository..."
if [ "$DELETE_ECR" = "yes" ]; then
    if aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION > /dev/null 2>&1; then
        aws ecr delete-repository --repository-name $REPO_NAME --force --region $REGION
        echo "  ✓ ECR repository deleted: $REPO_NAME"
    else
        echo "  (repository not found, skipping)"
    fi
else
    echo "  Keeping ECR repository (pass 'yes' as argument to delete)"
    echo "  To delete manually: aws ecr delete-repository --repository-name $REPO_NAME --force"
fi

# Delete configuration file
if [ -f "$CONFIG_FILE" ]; then
    rm -f "$CONFIG_FILE"
    echo ""
    echo "  ✓ Configuration file deleted"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ DELETION COMPLETE                                              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "All drift monitoring resources have been deleted."
echo ""
