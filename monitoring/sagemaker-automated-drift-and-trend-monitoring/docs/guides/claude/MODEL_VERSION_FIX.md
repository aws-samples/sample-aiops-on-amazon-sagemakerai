# Model Version Tracking Fix

## Problem

The `inference_responses` table was showing:
- `model_version = "unknown"`
- `mlflow_run_id = "unknown"`

This made it impossible to:
- Track which pipeline version produced which predictions
- Correlate drift with specific model versions
- Compare model performance across versions

## Root Cause

The SageMaker pipeline's `DeployEndpoint` step was **not passing** `model_version` or `mlflow_run_id` to the endpoint deployment Lambda.

### What Was Missing

**src/train_pipeline/pipeline.py** - `_create_deploy_step()` method:

```python
# OLD - Missing version info
inputs={
    "model_name": create_model_step.properties.ModelName,
    "endpoint_name": params['endpoint_name'],
    "memory_size_mb": params['endpoint_memory_size'],
    "max_concurrency": params['endpoint_max_concurrency'],
},
```

The Lambda `deploy_endpoint.py` expected these parameters but they weren't being passed:
- `model_version` - defaults to "v1.0" if not provided
- `mlflow_run_id` - defaults to "unknown" if not provided

## Fix Applied

### 1. Updated Pipeline Deploy Step

**src/train_pipeline/pipeline.py** - `_create_deploy_step()`:

```python
# NEW - Includes version tracking
inputs={
    "model_name": create_model_step.properties.ModelName,
    "endpoint_name": params['endpoint_name'],
    "memory_size_mb": params['endpoint_memory_size'],
    "max_concurrency": params['endpoint_max_concurrency'],
    "enable_athena_logging": params['enable_athena_logging'],
    "model_version": ExecutionVariables.PIPELINE_EXECUTION_ID,  # NEW
    "mlflow_run_id": "pipeline-run",  # NEW
},
```

### What Each Field Does

| Field | Value | Purpose |
|-------|-------|---------|
| `model_version` | `ExecutionVariables.PIPELINE_EXECUTION_ID` | Unique pipeline execution ID (e.g., `pipeline-20260323-123456`) |
| `mlflow_run_id` | `"pipeline-run"` | Identifies this as a pipeline-deployed model |
| `enable_athena_logging` | `params['enable_athena_logging']` | Passes logging config from pipeline parameters |

## How It Works

### Flow

```
1. Pipeline Execution Starts
   ↓
   PIPELINE_EXECUTION_ID = "fraud-detection-pipeline-20260323-143022"

2. Training Step
   ↓
   MLflow logs metrics with run_id

3. Deploy Step (Lambda)
   ↓
   Receives: model_version = "fraud-detection-pipeline-20260323-143022"
   Receives: mlflow_run_id = "pipeline-run"

4. Endpoint Created
   ↓
   Environment variables set:
   - MODEL_VERSION = "fraud-detection-pipeline-20260323-143022"
   - MLFLOW_RUN_ID = "pipeline-run"

5. Inference Handler
   ↓
   Logs to Athena with actual version numbers

6. inference_responses Table
   ↓
   model_version = "fraud-detection-pipeline-20260323-143022"
   mlflow_run_id = "pipeline-run"
```

## Verification

### After Next Pipeline Run

1. **Check Pipeline Execution**:
   ```python
   import boto3
   sm = boto3.client('sagemaker')

   # Get latest execution
   executions = sm.list_pipeline_executions(
       PipelineName='fraud-detection-pipeline',
       MaxResults=1
   )

   execution_arn = executions['PipelineExecutionSummaries'][0]['PipelineExecutionArn']
   execution_id = execution_arn.split('/')[-1]
   print(f"Pipeline Execution ID: {execution_id}")
   ```

2. **Check Endpoint Environment**:
   ```python
   # Get endpoint config
   endpoint = sm.describe_endpoint(EndpointName='fraud-detector-endpoint')
   config_name = endpoint['EndpointConfigName']

   config = sm.describe_endpoint_config(EndpointConfigName=config_name)
   env_vars = config['ProductionVariants'][0]['ModelDataDownloadTimeoutInSeconds']

   # Check container environment
   model_name = config['ProductionVariants'][0]['ModelName']
   model = sm.describe_model(ModelName=model_name)

   env = model['PrimaryContainer'].get('Environment', {})
   print(f"MODEL_VERSION: {env.get('MODEL_VERSION', 'NOT SET')}")
   print(f"MLFLOW_RUN_ID: {env.get('MLFLOW_RUN_ID', 'NOT SET')}")
   ```

3. **Check Inference Logs**:
   ```sql
   SELECT DISTINCT model_version, mlflow_run_id
   FROM fraud_detection.inference_responses
   ORDER BY request_timestamp DESC
   LIMIT 5;
   ```

   Should now show:
   ```
   model_version                                | mlflow_run_id
   ---------------------------------------------|---------------
   fraud-detection-pipeline-20260323-143022     | pipeline-run
   ```

## Benefits

✅ **Track Model Versions**: Each pipeline run creates a unique version identifier
✅ **Correlate Drift**: See which pipeline version's predictions are drifting
✅ **Compare Performance**: Compare metrics across pipeline executions
✅ **Audit Trail**: Know exactly which pipeline execution produced which predictions
✅ **QuickSight Visuals**: Feature drift by model version now shows actual versions

## Next Enhancement (Optional)

To get the actual MLflow run ID from the training step:

1. **Training Step**: Export MLflow run ID as output
2. **Deploy Step**: Use `training_step.properties.Outputs["mlflow_run_id"]`

This would replace `"pipeline-run"` with the actual MLflow run ID for deeper tracking.

## Rollout

### For New Pipelines
- ✅ Automatic - just run the pipeline
- New endpoints will have proper version tracking

### For Existing Endpoints
The endpoint needs to be **redeployed** to pick up the new environment variables:

**Option 1: Run Pipeline** (Recommended)
```python
# Run pipeline - it will update the existing endpoint
response = sagemaker_client.start_pipeline_execution(
    PipelineName='fraud-detection-pipeline',
    PipelineExecutionDisplayName=f'fraud-detection-pipeline-{timestamp}'
)
```

**Option 2: Manual Redeploy**
```python
# Use src/train_pipeline/deploy.py
python -m src.train_pipeline.deploy \\
    --model-name <model-name> \\
    --endpoint-name fraud-detector-endpoint \\
    --model-version fraud-detection-v1.0 \\
    --mlflow-run-id <actual-run-id>
```

## Verification Queries

### Check Version Distribution
```sql
SELECT
    model_version,
    COUNT(*) as prediction_count,
    MIN(request_timestamp) as first_prediction,
    MAX(request_timestamp) as last_prediction
FROM fraud_detection.inference_responses
GROUP BY model_version
ORDER BY MIN(request_timestamp) DESC;
```

### Check Drift by Version
```sql
SELECT
    m.model_version,
    AVG(m.drifted_columns_share) as avg_drift,
    COUNT(*) as monitoring_runs
FROM fraud_detection.monitoring_responses m
GROUP BY m.model_version
ORDER BY AVG(m.drifted_columns_share) DESC;
```

## Files Changed

- ✅ `src/train_pipeline/pipeline.py` - Added model_version and mlflow_run_id to deploy step inputs
- 📝 `docs/guides/claude/MODEL_VERSION_FIX.md` - This documentation

## Impact on QuickSight

After redeploying the endpoint and running inference:

**Sheet 3: Feature Drift Analysis** will now show:
- Actual pipeline execution IDs instead of "unknown"
- Can filter/compare by specific pipeline versions
- "Drift by Model Version Over Time" visual will be meaningful

**Sheet 4: Feature Drift Detail** will show:
- Which pipeline version had which feature drift
- Can track feature stability improvements across versions
