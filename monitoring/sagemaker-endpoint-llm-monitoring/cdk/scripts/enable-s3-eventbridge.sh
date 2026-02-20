#!/bin/bash
set -e

# Script to enable EventBridge notifications on S3 bucket

echo "========================================="
echo "Enable S3 EventBridge Notifications"
echo "========================================="
echo ""

# Get the bucket name from CDK outputs
echo "Fetching S3 bucket name from CloudFormation..."
BUCKET_NAME=$(aws cloudformation describe-stacks \
  --stack-name SageMakerInferenceMonitoringStack \
  --query 'Stacks[0].Outputs[?OutputKey==`DataCaptureBucketName`].OutputValue' \
  --output text)

if [ -z "$BUCKET_NAME" ]; then
    echo "ERROR: Could not find bucket name from stack outputs."
    echo "Please ensure the stack is deployed or specify bucket name manually:"
    echo ""
    echo "  export BUCKET_NAME=your-bucket-name"
    echo "  $0"
    exit 1
fi

echo "Found bucket: $BUCKET_NAME"
echo ""

# Check if EventBridge is already enabled
echo "Checking current notification configuration..."
CURRENT_CONFIG=$(aws s3api get-bucket-notification-configuration \
  --bucket $BUCKET_NAME 2>/dev/null || echo "{}")

if echo "$CURRENT_CONFIG" | grep -q "EventBridgeConfiguration"; then
    echo "✓ EventBridge notifications are already enabled on bucket $BUCKET_NAME"
    exit 0
fi

# Enable EventBridge notifications
echo "Enabling EventBridge notifications on bucket $BUCKET_NAME..."
aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration '{
    "EventBridgeConfiguration": {}
  }'

echo ""
echo "✓ Successfully enabled EventBridge notifications!"
echo ""
echo "The monitoring pipeline is now fully active."
echo "New .jsonl files in the data capture prefix will automatically trigger processing."
