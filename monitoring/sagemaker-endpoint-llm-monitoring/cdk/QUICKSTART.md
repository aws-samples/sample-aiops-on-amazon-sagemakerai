# Quick Start Guide

Get your SageMaker Inference Monitoring pipeline up and running in minutes.

## Prerequisites Checklist

- [ ] AWS CLI installed and configured
- [ ] Node.js 18+ installed
- [ ] Docker installed and running
- [ ] AWS CDK CLI installed (`npm install -g aws-cdk`)
- [ ] SageMaker endpoint with Data Capture enabled
- [ ] MLflow tracking server (SageMaker MLflow App) created

## Step-by-Step Setup

### 1. Configure Your Environment

Create a `.env` file in the `cdk/` directory by copying the example:

```bash
cd cdk
cp .env.example .env
```

Edit `.env` and update the **required** values:

```bash
# REQUIRED: Stack prefix (e.g., dev, prod, staging)
# This creates unique resource names to prevent conflicts
STACK_PREFIX=dev

# REQUIRED: MLflow tracking server ARN
MLFLOW_TRACKING_URI=arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-XXXXX

# REQUIRED: SageMaker endpoint name
SAGEMAKER_ENDPOINT_NAME=your-endpoint-name
```

**Smart Defaults** (optional overrides):
The following values use smart defaults based on your endpoint name and AWS account:

```bash
# Defaults to: sagemaker-{region}-{account}
# DATA_CAPTURE_S3_BUCKET=my-custom-bucket

# Defaults to: {endpoint-name}-data-capture/
# DATA_CAPTURE_S3_PREFIX=custom-prefix/

# Defaults to: sagemaker-endpoint-{endpoint-name}-monitoring
# MLFLOW_EXPERIMENT_NAME=my-custom-experiment

# Defaults to: Claude Sonnet 4
# BEDROCK_MODEL_ID=bedrock:/us.anthropic.claude-3-5-haiku-20241022-v1:0
```

### 2. Install Dependencies

```bash
# Install npm packages (including dotenv for .env support)
npm install
```

### 3. Deploy the Stack

```bash
# Run the deployment script
./scripts/deploy.sh
```

This will:
- Build TypeScript code
- Deploy CDK stack (Lambda, Step Functions, EventBridge rule, IAM roles)

⏱️ **Deployment takes ~5-10 minutes** (Lambda container image needs to be built and pushed to ECR)

### 4. Enable S3 EventBridge Notifications

```bash
# Run the helper script
./scripts/enable-s3-eventbridge.sh
```

Or manually:

```bash
BUCKET_NAME="sagemaker-us-west-2-123456789012"  # Your bucket

aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration '{"EventBridgeConfiguration": {}}'
```

### 5. Test the Pipeline

Find an existing data capture file:

```bash
# List data capture files
aws s3 ls s3://sagemaker-us-west-2-123456789012/llama-data-capture/ --recursive | grep .jsonl
```

Test with a specific file:

```bash
# Replace with your actual file path
./scripts/test-pipeline.sh "llama-data-capture/endpoint-name/AllTraffic/2024/01/15/10/capture-file.jsonl"
```

### 6. View Results

**Option A: MLflow UI (Recommended)**
1. Open SageMaker Studio
2. Navigate to MLflow app
3. Select your experiment (e.g., "sagemaker-endpoint-llama-3-1-8b-monitoring2")
4. View traces and evaluation scores

**Option B: CloudWatch Logs**
```bash
# View Lambda logs
aws logs tail /aws/lambda/SageMakerInferenceMonitoringStack-ProcessorLambda* --follow
```

**Option C: Python API**
```python
import mlflow

# Set tracking URI to your MLflow app
mlflow.set_tracking_uri("arn:aws:sagemaker:region:account:mlflow-app/app-id")

# Search traces
traces = mlflow.search_traces()
print(f"Total traces: {len(traces)}")

# Filter error traces
error_traces = mlflow.search_traces(
    filter_string="attributes.`status_code` >= '400'"
)
print(f"Error traces: {len(error_traces)}")
```

## Verification

After deploying, verify everything is working:

### 1. Check Lambda Function

```bash
aws lambda get-function \
  --function-name $(aws cloudformation describe-stacks \
    --stack-name SageMakerInferenceMonitoringStack \
    --query 'Stacks[0].Outputs[?OutputKey==`ProcessorLambdaArn`].OutputValue' \
    --output text | cut -d':' -f7)
```

Expected: Function exists and status is "Active"

### 2. Check Step Functions

```bash
aws stepfunctions describe-state-machine \
  --state-machine-arn $(aws cloudformation describe-stacks \
    --stack-name SageMakerInferenceMonitoringStack \
    --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' \
    --output text)
```

Expected: State machine exists and status is "ACTIVE"

### 3. Check EventBridge Rule

```bash
aws events list-rules --name-prefix SageMakerInferenceMonitoring
```

Expected: Rule exists and state is "ENABLED"

### 4. Trigger Real Inference

Make a request to your SageMaker endpoint:

```python
import boto3
import json

sagemaker_runtime = boto3.client('sagemaker-runtime')

response = sagemaker_runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps({
        "inputs": "What is machine learning?",
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    })
)

print(response)
```

Wait ~1-2 minutes for:
1. SageMaker to write data capture file to S3
2. EventBridge to trigger Step Functions
3. Step Functions to invoke Lambda
4. Lambda to process and log to MLflow

## Troubleshooting

### Issue: Deployment fails with "Docker not running"

**Solution**: Start Docker and retry deployment

```bash
# Check Docker status
docker ps

# If not running, start Docker Desktop or Docker daemon
```

### Issue: "No stacks found" when enabling S3 EventBridge

**Solution**: Ensure deployment completed successfully

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name SageMakerInferenceMonitoringStack
```

### Issue: No traces appearing in MLflow

**Solution**: Check Lambda execution logs

```bash
# View recent Lambda logs
aws logs tail /aws/lambda/SageMakerInferenceMonitoringStack-ProcessorLambda* --since 30m
```

Common causes:
- Incorrect MLflow tracking URI
- Lambda lacks IAM permissions
- Experiment name mismatch

### Issue: Step Functions fails with timeout

**Solution**: Reduce batch size or increase Lambda timeout

Edit [`lib/sagemaker-inference-monitoring-stack.ts`](lib/sagemaker-inference-monitoring-stack.ts):

```typescript
// Reduce batch size
itemBatcher: {
  maxItemsPerBatch: 5,  // Reduced from 10
}

// Increase Lambda timeout
timeout: cdk.Duration.minutes(15),  // Increased from 10
```

Redeploy:
```bash
cdk deploy
```

## Next Steps

- **Customize Evaluations**: Add custom scorers in [`lambda/handler.py`](lambda/handler.py)
- **Set Up Alerts**: Create CloudWatch alarms for error rates
- **Configure Sampling**: Process only a percentage of inferences to reduce costs
- **Dashboard**: Build QuickSight dashboard using MLflow data
- **Cost Optimization**: Adjust Lambda memory, use cheaper Bedrock models

## Clean Up

To remove all resources:

```bash
cdk destroy
```

⚠️ **Warning**: This does NOT delete:
- S3 bucket (your data capture files)
- MLflow traces (stored in MLflow app)
- CloudWatch log groups (if retention is set)

## Support

For help:
- Review [README.md](README.md) for detailed documentation
- Check CloudWatch logs for errors
- Review Step Functions execution history in AWS Console
- Verify IAM permissions

## Architecture Diagram

```
┌─────────────────┐
│  SageMaker      │
│  Endpoint       │
│  (Data Capture) │
└────────┬────────┘
         │ Writes .jsonl
         ▼
┌─────────────────┐
│  S3 Bucket      │
│  (Data Capture) │
└────────┬────────┘
         │ Triggers
         ▼
┌─────────────────┐
│  EventBridge    │
│  (S3 Events)    │
└────────┬────────┘
         │ Starts
         ▼
┌─────────────────┐
│  Step Functions │
│  (Distributed   │
│   Map - JSONL)  │
└────────┬────────┘
         │ Invokes
         ▼
┌─────────────────┐     ┌──────────────┐
│  Lambda         │────▶│  MLflow      │
│  (Process &     │     │  (Traces)    │
│   Evaluate)     │     └──────────────┘
└────────┬────────┘
         │ Calls
         ▼
┌─────────────────┐
│  Amazon Bedrock │
│  (GenAI Evals)  │
└─────────────────┘
```

---

**🎉 Congratulations!** Your automated SageMaker inference monitoring pipeline is now live!
