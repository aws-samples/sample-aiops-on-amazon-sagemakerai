#!/bin/bash
set -e

# Script to test the monitoring pipeline with a sample data capture file

echo "========================================="
echo "Test SageMaker Inference Monitoring Pipeline"
echo "========================================="
echo ""

# Check for required argument
if [ -z "$1" ]; then
    echo "Usage: $0 <s3-key-to-jsonl-file>"
    echo ""
    echo "Example:"
    echo "  $0 llama-data-capture/endpoint-name/AllTraffic/2024/01/15/10/capture-file.jsonl"
    echo ""
    exit 1
fi

S3_KEY=$1

# Get bucket name and state machine ARN from CloudFormation
echo "Fetching configuration from CloudFormation..."

BUCKET_NAME=$(aws cloudformation describe-stacks \
  --stack-name SageMakerInferenceMonitoringStack \
  --query 'Stacks[0].Outputs[?OutputKey==`DataCaptureBucketName`].OutputValue' \
  --output text)

STATE_MACHINE_ARN=$(aws cloudformation describe-stacks \
  --stack-name SageMakerInferenceMonitoringStack \
  --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' \
  --output text)

if [ -z "$BUCKET_NAME" ] || [ -z "$STATE_MACHINE_ARN" ]; then
    echo "ERROR: Could not retrieve stack outputs. Is the stack deployed?"
    exit 1
fi

echo "Bucket: $BUCKET_NAME"
echo "S3 Key: $S3_KEY"
echo "State Machine: $STATE_MACHINE_ARN"
echo ""

# Verify the S3 file exists
echo "Verifying S3 file exists..."
if aws s3 ls "s3://$BUCKET_NAME/$S3_KEY" > /dev/null 2>&1; then
    echo "✓ File found"
else
    echo "ERROR: File not found: s3://$BUCKET_NAME/$S3_KEY"
    exit 1
fi

# Create input for Step Functions
INPUT_JSON=$(cat <<EOF
{
  "detail": {
    "bucket": {
      "name": "$BUCKET_NAME"
    },
    "object": {
      "key": "$S3_KEY"
    }
  }
}
EOF
)

echo ""
echo "Starting Step Functions execution..."
EXECUTION_ARN=$(aws stepfunctions start-execution \
  --state-machine-arn "$STATE_MACHINE_ARN" \
  --input "$INPUT_JSON" \
  --query 'executionArn' \
  --output text)

echo "✓ Execution started: $EXECUTION_ARN"
echo ""
echo "Monitoring execution status..."
echo "(Press Ctrl+C to stop monitoring, execution will continue)"
echo ""

# Monitor execution
while true; do
    STATUS=$(aws stepfunctions describe-execution \
      --execution-arn "$EXECUTION_ARN" \
      --query 'status' \
      --output text)

    echo "Status: $STATUS"

    if [ "$STATUS" = "SUCCEEDED" ]; then
        echo ""
        echo "✓ Execution completed successfully!"

        # Show output
        echo ""
        echo "Execution output:"
        aws stepfunctions describe-execution \
          --execution-arn "$EXECUTION_ARN" \
          --query 'output' \
          --output text | jq '.' || true

        break
    elif [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "TIMED_OUT" ] || [ "$STATUS" = "ABORTED" ]; then
        echo ""
        echo "✗ Execution failed with status: $STATUS"

        # Show error
        echo ""
        echo "Error details:"
        aws stepfunctions describe-execution \
          --execution-arn "$EXECUTION_ARN" \
          --query '{cause: cause, error: error}' \
          --output json | jq '.'

        exit 1
    fi

    sleep 5
done

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo ""
echo "View results:"
echo "  - CloudWatch Logs: /aws/lambda/SageMakerInferenceMonitoringStack-ProcessorLambda*"
echo "  - Step Functions Console: https://console.aws.amazon.com/states/home?region=$(aws configure get region)#/executions/details/$EXECUTION_ARN"
echo "  - MLflow UI: Access via SageMaker Studio"
echo ""
