# Model Registry Version Tracking - Complete Fix

## Problem

The `inference_responses` table was showing:
- `model_version = "pipeline"` (hardcoded literal)
- `mlflow_run_id = "pipeline"` (hardcoded literal)

**User Requirement**: Use SageMaker Model Registry version numbers (v1, v2, v3) that are already tracked in the model packages.

## Solution Architecture

### Data Flow

```
1. Training Pipeline Execution
   ↓
2. RegisterModel Step
   ↓
   Creates Model Package in Model Registry
   ARN: arn:aws:sagemaker:region:account:model-package/fraud-detection/3
                                                                        ↑
                                                               Version Number
3. CreateModelStep
   ↓
   Creates Model with environment: MODEL_VERSION='pipeline' (to be updated)

4. DeployEndpoint Lambda
   ↓
   Receives: model_package_arn from RegisterModel step
   Extracts: version number "3" from ARN
   Formats: "v3"
   ↓
   Describes existing Model
   Updates environment: MODEL_VERSION='v3', MLFLOW_RUN_ID='pipeline'
   Creates new Model with updated environment
   ↓
   Creates Endpoint using new Model

5. Inference Handler
   ↓
   Reads: os.environ['MODEL_VERSION'] → "v3"
   Reads: os.environ['MLFLOW_RUN_ID'] → "pipeline"
   ↓
   Logs to Athena with actual version numbers

6. inference_responses Table
   ↓
   model_version = "v3"
   mlflow_run_id = "pipeline"
```

## Changes Made

### 1. Pipeline: Pass Model Package ARN to Lambda

**File**: `src/train_pipeline/pipeline.py`

**Line 709**: Added `model_package_arn` parameter to deploy Lambda inputs

```python
inputs={
    "model_name": create_model_step.properties.ModelName,
    "endpoint_name": params['endpoint_name'],
    "memory_size_mb": params['endpoint_memory_size'],
    "max_concurrency": params['endpoint_max_concurrency'],
    "enable_athena_logging": params['enable_athena_logging'],
    "model_package_arn": register_step.properties.ModelPackageArn,  # NEW
    "mlflow_run_id": "pipeline",
},
```

**What it does**: Passes the Model Registry ARN from the RegisterModel step to the deployment Lambda.

**Example ARN**: `arn:aws:sagemaker:us-east-1:123456789012:model-package/fraud-detection/3`

---

### 2. Lambda: Extract Version and Update Model Environment

**File**: `src/train_pipeline/pipeline_steps/lambda_deploy_endpoint.py`

**Added Logic** (lines 29-58):

```python
# Extract version from Model Package ARN
model_package_arn = event.get('model_package_arn')
mlflow_run_id = event.get('mlflow_run_id', 'pipeline')

model_version = 'v1'  # Default
if model_package_arn:
    try:
        # Split ARN and get the last segment (version number)
        version_number = model_package_arn.split('/')[-1]
        model_version = f"v{version_number}"
        logger.info(f"Extracted model version: {model_version} from {model_package_arn}")
    except Exception as e:
        logger.warning(f"Could not extract version from ARN: {e}")

# Update Model environment variables with version info
try:
    model_desc = sagemaker_client.describe_model(ModelName=model_name)
    current_env = model_desc['PrimaryContainer'].get('Environment', {})

    # Update with version info
    current_env['MODEL_VERSION'] = model_version
    current_env['MLFLOW_RUN_ID'] = mlflow_run_id

    # Create new model with updated env vars (models are immutable)
    new_model_name = f"{model_name}-{int(time.time())}"
    sagemaker_client.create_model(
        ModelName=new_model_name,
        PrimaryContainer={
            'Image': model_desc['PrimaryContainer']['Image'],
            'ModelDataUrl': model_desc['PrimaryContainer']['ModelDataUrl'],
            'Environment': current_env
        },
        ExecutionRoleArn=model_desc['ExecutionRoleArn']
    )

    # Use the new model for endpoint
    model_name = new_model_name

except Exception as e:
    logger.warning(f"Could not update model environment: {e}")
```

**What it does**:
1. Extracts version number from ARN (e.g., "3" → "v3")
2. Describes the existing Model to get its container configuration
3. Updates environment variables with MODEL_VERSION and MLFLOW_RUN_ID
4. Creates a new Model with updated environment (SageMaker models are immutable)
5. Uses the new Model for endpoint deployment

**Why create a new model**: SageMaker Model resources are immutable. To change environment variables, you must create a new Model with the updated configuration.

## Version Number Format

| Model Package Version | Extracted Value | Final MODEL_VERSION |
|----------------------|-----------------|---------------------|
| 1                    | "1"             | "v1"                |
| 2                    | "2"             | "v2"                |
| 3                    | "3"             | "v3"                |
| 15                   | "15"            | "v15"               |

## Testing the Fix

### 1. Run the Pipeline

```python
import boto3
from src.train_pipeline.pipeline import create_fraud_detection_pipeline

# Create and upsert pipeline
pipeline_builder = create_fraud_detection_pipeline(
    pipeline_name='fraud-detection-pipeline',
    region='us-east-1'
)

result = pipeline_builder.upsert_pipeline(include_deployment=True)
print(f"Pipeline ARN: {result['pipeline_arn']}")

# Start execution
exec_result = pipeline_builder.start_execution()
print(f"Execution ARN: {exec_result['execution_arn']}")
```

### 2. Wait for Pipeline Completion

Monitor in SageMaker Console → Pipelines → fraud-detection-pipeline

Expected flow:
- PreprocessData → TrainModel → EvaluateModel → CheckModelQuality
- → RegisterModel (creates version 1, 2, 3, etc.)
- → CreateModel → DeployEndpoint (extracts version)
- → TestInference

### 3. Verify Model Package Version

```python
import boto3

sm = boto3.client('sagemaker')

# List model packages
response = sm.list_model_packages(
    ModelPackageGroupName='fraud-detection',
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)

latest_package = response['ModelPackageSummaryList'][0]
print(f"Latest Model Package ARN: {latest_package['ModelPackageArn']}")
print(f"Version: {latest_package['ModelPackageVersion']}")

# Example output:
# ARN: arn:aws:sagemaker:us-east-1:123456:model-package/fraud-detection/3
# Version: 3
```

### 4. Verify Endpoint Environment Variables

```python
# Get endpoint
endpoint = sm.describe_endpoint(EndpointName='fraud-detector')
config_name = endpoint['EndpointConfigName']

# Get endpoint config
config = sm.describe_endpoint_config(EndpointConfigName=config_name)
model_name = config['ProductionVariants'][0]['ModelName']

# Get model environment
model = sm.describe_model(ModelName=model_name)
env = model['PrimaryContainer'].get('Environment', {})

print(f"MODEL_VERSION: {env.get('MODEL_VERSION')}")
print(f"MLFLOW_RUN_ID: {env.get('MLFLOW_RUN_ID')}")

# Expected output:
# MODEL_VERSION: v3
# MLFLOW_RUN_ID: pipeline
```

### 5. Verify Inference Logs in Athena

```sql
SELECT DISTINCT
    model_version,
    mlflow_run_id,
    COUNT(*) as prediction_count,
    MIN(request_timestamp) as first_prediction,
    MAX(request_timestamp) as last_prediction
FROM fraud_detection.inference_responses
GROUP BY model_version, mlflow_run_id
ORDER BY MIN(request_timestamp) DESC;
```

**Expected Result**:
```
model_version | mlflow_run_id | prediction_count | first_prediction       | last_prediction
--------------|---------------|------------------|------------------------|------------------------
v3            | pipeline      | 150              | 2026-03-23 10:15:00    | 2026-03-23 10:30:00
v2            | pipeline      | 200              | 2026-03-22 14:20:00    | 2026-03-22 14:45:00
v1            | pipeline      | 100              | 2026-03-21 09:00:00    | 2026-03-21 09:15:00
```

### 6. Test Drift Monitoring by Version

```sql
SELECT
    m.model_version,
    AVG(m.drifted_columns_share) as avg_drift_percentage,
    AVG(m.current_roc_auc) as avg_current_roc_auc,
    COUNT(*) as monitoring_runs
FROM fraud_detection.monitoring_responses m
GROUP BY m.model_version
ORDER BY m.model_version DESC;
```

**Use Case**: Compare drift across model versions to see if newer models handle distribution shifts better.

## QuickSight Dashboard Impact

After the fix, your QuickSight dashboards will show:

### Sheet 3: Feature Drift Analysis
- **Visual 15: Drift by Model Version Over Time**
  - X-axis: Date
  - Y-axis: Drifted Columns Share
  - Color: model_version (now shows v1, v2, v3 instead of "pipeline")

- **Visual 16: Model Version Distribution**
  - Shows which versions are actively serving predictions
  - Can compare v1 vs v2 vs v3 performance

### Sheet 4: Feature Drift Detail (if implemented)
- **Filter by Model Version**: Select v1, v2, or v3 to see feature-level drift for specific versions
- **Compare Features Across Versions**: Track if specific features drift less in newer models

## Benefits

✅ **Track Model Evolution**: See which pipeline execution produced which predictions
✅ **Compare Model Versions**: Analyze drift and performance across v1, v2, v3
✅ **Model Governance**: Clear audit trail of model versions in production
✅ **Drift Analysis**: Correlate drift with specific model versions
✅ **Automatic Versioning**: Uses SageMaker Model Registry's built-in versioning

## Rollout

### For New Pipeline Executions
- ✅ Automatic - just run the pipeline
- Each execution increments the model package version
- Version automatically flows to inference logs

### For Existing Endpoints
The endpoint needs to be **redeployed** by running the pipeline:

```python
# Run pipeline - it will update the existing endpoint with new version
from src.train_pipeline.pipeline import create_fraud_detection_pipeline

pipeline_builder = create_fraud_detection_pipeline()
exec_result = pipeline_builder.start_execution()
print(f"Execution ARN: {exec_result['execution_arn']}")
```

**What happens**:
1. Pipeline creates new model package (e.g., version 4)
2. DeployEndpoint Lambda extracts "v4"
3. Creates new Model with MODEL_VERSION="v4"
4. Updates endpoint to use new Model
5. Inference logs now show model_version="v4"

## Model Package Group

The Model Registry uses **Model Package Groups** to organize versions:

**Group Name**: `fraud-detection` (from `MLFLOW_MODEL_NAME` in config)

**Version Sequence**:
- First pipeline run → version 1
- Second pipeline run → version 2
- Third pipeline run → version 3
- And so on...

**View in Console**: SageMaker → Model Registry → fraud-detection

## Files Changed

1. ✅ `src/train_pipeline/pipeline.py` (line 709)
   - Pass model_package_arn to deploy Lambda

2. ✅ `src/train_pipeline/pipeline_steps/lambda_deploy_endpoint.py` (lines 29-58)
   - Extract version from ARN
   - Update Model environment variables

3. 📝 `docs/guides/claude/MODEL_REGISTRY_VERSION_FIX.md` (this file)
   - Complete documentation

## Troubleshooting

### Version still shows "pipeline" in Athena

**Check 1**: Verify pipeline passed model_package_arn to Lambda
```python
# Look at Lambda CloudWatch logs
# Search for: "Extracted model version: v3"
```

**Check 2**: Verify Model has updated environment
```python
# Follow "Verify Endpoint Environment Variables" section above
```

**Check 3**: Re-run pipeline to update endpoint
```python
# The existing endpoint needs to be updated with a new deployment
pipeline_builder.start_execution()
```

### Lambda fails with "Could not update model environment"

**Cause**: Lambda role may not have `sagemaker:CreateModel` permission

**Fix**: Add to Lambda execution role policy:
```json
{
    "Effect": "Allow",
    "Action": [
        "sagemaker:DescribeModel",
        "sagemaker:CreateModel"
    ],
    "Resource": "arn:aws:sagemaker:*:*:model/*"
}
```

### Version shows "v1" for all runs

**Cause**: Model package group doesn't exist or versions aren't incrementing

**Check**: Verify model package group exists
```python
sm.list_model_package_groups(NameContains='fraud-detection')
```

**Fix**: Ensure RegisterModel step is running successfully in the pipeline

## Next Enhancements (Optional)

### 1. Add Pipeline Execution ID

For even more granular tracking, also pass the pipeline execution ID:

```python
# In pipeline.py, _create_deploy_step
inputs={
    # ... existing params ...
    "pipeline_execution_id": ExecutionVariables.PIPELINE_EXECUTION_ID,
}

# In lambda_deploy_endpoint.py
pipeline_exec_id = event.get('pipeline_execution_id', 'unknown')
current_env['PIPELINE_EXECUTION_ID'] = pipeline_exec_id
```

### 2. Get Actual MLflow Run ID from Training

Currently `mlflow_run_id` is hardcoded to "pipeline". To get the actual MLflow run ID:

1. In `train.py`, export run_id as output
2. In pipeline, pass `training_step.properties.Outputs["mlflow_run_id"]` to deploy step

### 3. Add Model Approval Workflow

Change from auto-approval to manual approval:

```python
# In pipeline parameters
'model_approval_status': ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"  # Changed from "Approved"
),
```

Then only approved models get deployed to production.

## Summary

This fix ensures that SageMaker Model Registry version numbers (v1, v2, v3...) automatically flow through to inference logs and QuickSight dashboards, enabling version-based drift analysis and model performance comparison.

**Key Insight**: The version is extracted from the Model Package ARN created by the RegisterModel step, so it uses SageMaker's native versioning system rather than requiring manual version management.
