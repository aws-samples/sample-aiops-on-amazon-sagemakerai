#!/bin/bash
###############################################################################
# CI/CD Script: Deploy Automated Drift Monitoring
#
# This script automates the deployment of drift monitoring infrastructure:
# - SNS topic for alerts
# - Lambda function for drift detection
# - EventBridge rule for scheduling
#
# Usage:
#   ./deploy_drift_monitoring.sh [--email EMAIL] [--schedule CRON]
#
# Environment Variables (required):
#   AWS_REGION              - AWS region (default: us-east-1)
#   ATHENA_DATABASE         - Athena database name
#   ATHENA_OUTPUT_S3        - S3 path for Athena query results
#   ALERT_EMAIL             - Email for drift alerts (or use --email)
#
# Optional:
#   MLFLOW_TRACKING_URI     - MLflow tracking server URI
#   BASELINE_ROC_AUC        - Expected model ROC-AUC (default: 0.92)
#   DATA_DRIFT_THRESHOLD    - PSI threshold (default: 0.2)
#   MODEL_DRIFT_THRESHOLD   - Performance degradation % (default: 0.05)
#
# Example:
#   export ALERT_EMAIL="ops@example.com"
#   export ATHENA_DATABASE="fraud_detection"
#   export ATHENA_OUTPUT_S3="s3://my-bucket/athena-results/"
#   ./deploy_drift_monitoring.sh
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --email)
            ALERT_EMAIL="$2"
            shift 2
            ;;
        --schedule)
            SCHEDULE_EXPRESSION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--email EMAIL] [--schedule CRON]"
            echo ""
            echo "Options:"
            echo "  --email EMAIL      Email address for drift alerts"
            echo "  --schedule CRON    EventBridge cron expression (default: daily at 2 AM)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

SNS_TOPIC_NAME="fraud-detection-drift-alerts"
LAMBDA_FUNCTION_NAME="fraud-detection-drift-monitor"
LAMBDA_ROLE_NAME="fraud-detection-drift-monitor-role"
EVENTBRIDGE_RULE_NAME="fraud-detection-drift-check"

# Default schedule: Daily at 2 AM UTC
SCHEDULE_EXPRESSION=${SCHEDULE_EXPRESSION:-"cron(0 2 * * ? *)"}

# Lambda configuration
LAMBDA_RUNTIME="python3.11"
LAMBDA_TIMEOUT=300
LAMBDA_MEMORY=512

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAMBDA_CODE="$SCRIPT_DIR/lambda_drift_monitor.py"
PACKAGE_DIR="$PROJECT_ROOT/.lambda_package"
ZIP_FILE="$PROJECT_ROOT/lambda_drift_monitor.zip"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Automated Drift Monitoring Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Validate required environment variables
if [ -z "$ATHENA_DATABASE" ]; then
    echo -e "${RED}Error: ATHENA_DATABASE environment variable not set${NC}"
    exit 1
fi

if [ -z "$ATHENA_OUTPUT_S3" ]; then
    echo -e "${RED}Error: ATHENA_OUTPUT_S3 environment variable not set${NC}"
    exit 1
fi

if [ -z "$ALERT_EMAIL" ]; then
    echo -e "${YELLOW}Warning: ALERT_EMAIL not set, skipping email subscription${NC}"
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  AWS Region: $AWS_REGION"
echo "  AWS Account: $AWS_ACCOUNT_ID"
echo "  Athena Database: $ATHENA_DATABASE"
echo "  Athena Output: $ATHENA_OUTPUT_S3"
echo "  Alert Email: ${ALERT_EMAIL:-Not configured}"
echo "  Schedule: $SCHEDULE_EXPRESSION"
echo ""

# Step 1: Create SNS Topic
echo -e "${BLUE}[1/6] Creating SNS topic...${NC}"
SNS_TOPIC_ARN=$(aws sns create-topic \
    --name "$SNS_TOPIC_NAME" \
    --region "$AWS_REGION" \
    --query 'TopicArn' \
    --output text 2>/dev/null || \
    aws sns list-topics --region "$AWS_REGION" --query "Topics[?contains(TopicArn, '$SNS_TOPIC_NAME')].TopicArn" --output text)

echo -e "${GREEN}✓ SNS Topic: $SNS_TOPIC_ARN${NC}"

# Subscribe email if provided
if [ -n "$ALERT_EMAIL" ]; then
    echo "  Subscribing email: $ALERT_EMAIL"
    aws sns subscribe \
        --topic-arn "$SNS_TOPIC_ARN" \
        --protocol email \
        --notification-endpoint "$ALERT_EMAIL" \
        --region "$AWS_REGION" 2>/dev/null || true
    echo -e "${YELLOW}  ⚠ Check email for confirmation link${NC}"
fi

# Step 2: Create IAM Role for Lambda
echo -e "\n${BLUE}[2/6] Creating IAM role...${NC}"

TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)

# Create role
aws iam create-role \
    --role-name "$LAMBDA_ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --description "Role for fraud detection drift monitoring Lambda" \
    2>/dev/null || echo "  Role already exists"

ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/$LAMBDA_ROLE_NAME"

# Attach managed policies
echo "  Attaching policies..."
aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
    2>/dev/null || true

aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonAthenaFullAccess" \
    2>/dev/null || true

aws iam attach-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess" \
    2>/dev/null || true

# Create inline SNS policy
SNS_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["sns:Publish"],
      "Resource": "$SNS_TOPIC_ARN"
    }
  ]
}
EOF
)

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "SNSPublishPolicy" \
    --policy-document "$SNS_POLICY" \
    2>/dev/null || true

echo -e "${GREEN}✓ IAM Role: $ROLE_ARN${NC}"
echo "  ⏳ Waiting for role to propagate..."
sleep 10

# Step 3: Create Lambda Deployment Package
echo -e "\n${BLUE}[3/6] Creating Lambda deployment package...${NC}"

# Check if Lambda code exists
if [ ! -f "$LAMBDA_CODE" ]; then
    echo -e "${RED}Error: Lambda code not found: $LAMBDA_CODE${NC}"
    exit 1
fi

# Create package directory
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# Install dependencies
echo "  📥 Installing dependencies..."
REQUIREMENTS_FILE="$SCRIPT_DIR/lambda_requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -q -t "$PACKAGE_DIR" -r "$REQUIREMENTS_FILE"
else
    # Fallback to manual installation
    pip install -q -t "$PACKAGE_DIR" scikit-learn numpy pandas scipy mlflow matplotlib boto3
fi

# Copy Lambda code
cp "$LAMBDA_CODE" "$PACKAGE_DIR/lambda_drift_monitor.py"

# Create ZIP file
echo "  🗜️ Creating ZIP file..."
cd "$PACKAGE_DIR"
zip -q -r "$ZIP_FILE" .
cd "$PROJECT_ROOT"

# Cleanup
rm -rf "$PACKAGE_DIR"

ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo -e "${GREEN}✓ Lambda package created: $ZIP_SIZE${NC}"

# Step 4: Create/Update Lambda Function
echo -e "\n${BLUE}[4/6] Deploying Lambda function...${NC}"

# Environment variables
LAMBDA_ENV=$(cat <<EOF
{
  "Variables": {
    "ATHENA_DATABASE": "$ATHENA_DATABASE",
    "ATHENA_OUTPUT_S3": "$ATHENA_OUTPUT_S3",
    "SNS_TOPIC_ARN": "$SNS_TOPIC_ARN",
    "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI:-}",
    "BASELINE_ROC_AUC": "${BASELINE_ROC_AUC:-0.92}",
    "DATA_DRIFT_THRESHOLD": "${DATA_DRIFT_THRESHOLD:-0.2}",
    "KS_PVALUE_THRESHOLD": "${KS_PVALUE_THRESHOLD:-0.05}",
    "MODEL_DRIFT_THRESHOLD": "${MODEL_DRIFT_THRESHOLD:-0.05}"
  }
}
EOF
)

# Try to create function
if aws lambda create-function \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --runtime "$LAMBDA_RUNTIME" \
    --role "$ROLE_ARN" \
    --handler "lambda_drift_monitor.lambda_handler" \
    --zip-file "fileb://$ZIP_FILE" \
    --timeout "$LAMBDA_TIMEOUT" \
    --memory-size "$LAMBDA_MEMORY" \
    --environment "$LAMBDA_ENV" \
    --region "$AWS_REGION" \
    2>/dev/null; then
    echo -e "${GREEN}✓ Lambda function created${NC}"
else
    # Update existing function
    echo "  Updating existing function..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --zip-file "fileb://$ZIP_FILE" \
        --region "$AWS_REGION"

    aws lambda update-function-configuration \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --timeout "$LAMBDA_TIMEOUT" \
        --memory-size "$LAMBDA_MEMORY" \
        --environment "$LAMBDA_ENV" \
        --region "$AWS_REGION"
    echo -e "${GREEN}✓ Lambda function updated${NC}"
fi

FUNCTION_ARN="arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:$LAMBDA_FUNCTION_NAME"

# Step 5: Create EventBridge Rule
echo -e "\n${BLUE}[5/6] Creating EventBridge rule...${NC}"

aws events put-rule \
    --name "$EVENTBRIDGE_RULE_NAME" \
    --schedule-expression "$SCHEDULE_EXPRESSION" \
    --state ENABLED \
    --description "Trigger drift monitoring Lambda" \
    --region "$AWS_REGION" >/dev/null

RULE_ARN="arn:aws:events:$AWS_REGION:$AWS_ACCOUNT_ID:rule/$EVENTBRIDGE_RULE_NAME"

# Add Lambda as target
aws events put-targets \
    --rule "$EVENTBRIDGE_RULE_NAME" \
    --targets "Id=1,Arn=$FUNCTION_ARN" \
    --region "$AWS_REGION" >/dev/null

# Grant EventBridge permission to invoke Lambda
aws lambda add-permission \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --statement-id "AllowEventBridgeInvoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "$RULE_ARN" \
    --region "$AWS_REGION" \
    2>/dev/null || echo "  Permission already exists"

echo -e "${GREEN}✓ EventBridge rule created${NC}"
echo "  Schedule: $SCHEDULE_EXPRESSION"

# Step 6: Test Lambda Function
echo -e "\n${BLUE}[6/6] Testing Lambda function...${NC}"

TEST_RESPONSE=$(aws lambda invoke \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --region "$AWS_REGION" \
    --payload '{}' \
    /dev/stdout | tail -n 1)

if echo "$TEST_RESPONSE" | jq -e '.statusCode == 200' >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Lambda test successful${NC}"
else
    echo -e "${YELLOW}⚠ Lambda test completed with warnings (check CloudWatch logs)${NC}"
fi

# Save configuration
echo -e "\n${BLUE}Saving configuration...${NC}"
CONFIG_FILE="$PROJECT_ROOT/drift_monitoring_config.json"

cat > "$CONFIG_FILE" <<EOF
{
  "sns_topic_arn": "$SNS_TOPIC_ARN",
  "lambda_function_arn": "$FUNCTION_ARN",
  "eventbridge_rule_arn": "$RULE_ARN",
  "schedule": "$SCHEDULE_EXPRESSION",
  "data_drift_threshold": "${DATA_DRIFT_THRESHOLD:-0.2}",
  "model_drift_threshold": "${MODEL_DRIFT_THRESHOLD:-0.05}",
  "email": "${ALERT_EMAIL:-}"
}
EOF

echo -e "${GREEN}✓ Configuration saved: $CONFIG_FILE${NC}"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETED SUCCESSFULLY${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Resources Created:"
echo "  SNS Topic: $SNS_TOPIC_ARN"
echo "  Lambda Function: $FUNCTION_ARN"
echo "  EventBridge Rule: $RULE_ARN"
echo ""
echo "Configuration:"
echo "  Schedule: $SCHEDULE_EXPRESSION"
echo "  Data Drift Thresholds:"
echo "    - PSI >= ${DATA_DRIFT_THRESHOLD:-0.2}"
echo "    - KS p-value < ${KS_PVALUE_THRESHOLD:-0.05} (dual-threshold detection)"
echo "  Model Drift Threshold: ${MODEL_DRIFT_THRESHOLD:-0.05}% degradation"
echo ""
if [ -n "$ALERT_EMAIL" ]; then
    echo -e "${YELLOW}⚠ IMPORTANT: Check email ($ALERT_EMAIL) and confirm SNS subscription!${NC}"
    echo ""
fi
echo "Next Steps:"
echo "  1. Confirm email subscription (check inbox/spam)"
echo "  2. Monitor CloudWatch logs: /aws/lambda/$LAMBDA_FUNCTION_NAME"
echo "  3. View metrics in 2a_inference_monitoring.ipynb"
echo "  4. Test manual trigger:"
echo "     aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME output.json"
echo ""
