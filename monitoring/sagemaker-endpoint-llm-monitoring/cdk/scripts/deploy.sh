#!/bin/bash
set -e

# Deployment script for SageMaker Inference Monitoring CDK stack

echo "========================================="
echo "SageMaker Inference Monitoring Deployment"
echo "========================================="
echo ""

# Check for required environment variables
if [ -z "$MLFLOW_TRACKING_URI" ]; then
    echo "WARNING: MLFLOW_TRACKING_URI not set. Please set it or update bin/sagemaker-inference-monitoring-cdk.ts"
fi

if [ -z "$SAGEMAKER_ENDPOINT_NAME" ]; then
    echo "WARNING: SAGEMAKER_ENDPOINT_NAME not set. Please set it or update bin/sagemaker-inference-monitoring-cdk.ts"
fi

echo "Step 1: Installing dependencies..."
npm install

echo ""
echo "Step 2: Building TypeScript..."
npm run build

echo ""
echo "Step 3: Synthesizing CDK stack..."
npx cdk synth

echo ""
echo "Step 4: Deploying CDK stack..."
npx cdk deploy --require-approval never

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "IMPORTANT: You must enable EventBridge notifications on your S3 bucket."
echo "Run the following command:"
echo ""
echo "  ./scripts/enable-s3-eventbridge.sh"
echo ""
echo "Or manually run:"
echo ""
echo '  aws s3api put-bucket-notification-configuration \'
echo '    --bucket YOUR_BUCKET_NAME \'
echo "    --notification-configuration '{\"EventBridgeConfiguration\": {}}'"
echo ""
