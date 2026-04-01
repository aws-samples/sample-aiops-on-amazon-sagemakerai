# Version Management Guide

## Overview

This document describes the comprehensive version management system for the fraud detection ML pipeline. Version tracking ensures traceability from model training through deployment to inference, enabling reproducibility, debugging, and governance.

## Version Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ML PIPELINE VERSION FLOW                         │
└─────────────────────────────────────────────────────────────────────────┘

1. TRAINING
   ┌──────────────────────────────┐
   │ XGBoost Training             │
   │ - src/train_pipeline/        │
   │   pipeline_steps/train.py    │
   └──────────┬───────────────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │ MLflow Registration          │
   │ - Auto-increment version     │
   │ - Version: 1, 2, 3...        │
   │ - Stage: Staging/Production  │
   └──────────┬───────────────────┘
              │
              ▼
2. REGISTRATION
   ┌──────────────────────────────┐
   │ SageMaker Model Registry     │
   │ - Model Package ARN          │
   │ - Version extracted from ARN │
   │ - Format: "v1", "v2"...      │
   └──────────┬───────────────────┘
              │
              ▼
3. DEPLOYMENT
   ┌──────────────────────────────┐
   │ Lambda Deploy Function       │
   │ - Extract version from ARN   │
   │ - Set MODEL_VERSION env var  │
   │ - Set MLFLOW_RUN_ID env var  │
   └──────────┬───────────────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │ SageMaker Endpoint           │
   │ - Environment variables:     │
   │   * MODEL_VERSION            │
   │   * MLFLOW_RUN_ID            │
   │   * ENDPOINT_NAME            │
   └──────────┬───────────────────┘
              │
              ▼
4. INFERENCE
   ┌──────────────────────────────┐
   │ Inference Handler            │
   │ - Reads env variables        │
   │ - Includes in response:      │
   │   metadata.model_version     │
   │   metadata.mlflow_run_id     │
   │   metadata.endpoint_name     │
   └──────────┬───────────────────┘
              │
              ▼
5. LOGGING
   ┌──────────────────────────────┐
   │ Athena Tables                │
   │ - inference_responses        │
   │ - monitoring_responses       │
   │ - Stores version with each   │
   │   inference for tracking     │
   └──────────────────────────────┘
```

## Version Identifiers

### 1. MLflow Model Version
- **Format**: Integer (1, 2, 3, ...)
- **Source**: Auto-incremented by MLflow Registry
- **Location**: `src/train_pipeline/pipeline_steps/train.py:677`
- **Query Method**:
  ```python
  from mlflow.tracking import MlflowClient
  client = MlflowClient()
  model_versions = client.search_model_versions(f"name='{model_name}'")
  latest_version = max([int(mv.version) for mv in model_versions])
  ```

### 2. SageMaker Model Version
- **Format**: String ("v1", "v2", "v3", ...)
- **Source**: Extracted from Model Package ARN
- **Location**: `src/train_pipeline/pipeline_steps/lambda_deploy_endpoint.py:42`
- **Extraction Logic**:
  ```python
  version_number = model_package_arn.split('/')[-1]  # Get last segment
  model_version = f"v{version_number}"
  ```

### 3. Inference Response Version
- **Format**: String ("v1", "v2", "v3", ...)
- **Source**: Environment variable `MODEL_VERSION`
- **Location**: Response metadata
- **Example Response**:
  ```json
  {
    "predictions": [0, 1],
    "probabilities": {
      "non_fraud": [0.95, 0.12],
      "fraud": [0.05, 0.88]
    },
    "metadata": {
      "model_version": "v2",
      "mlflow_run_id": "abc123def456",
      "endpoint_name": "fraud-detection-endpoint"
    }
  }
  ```

### 4. Athena Logged Version
- **Format**: String
- **Source**: Copied from inference response
- **Location**: `inference_responses` table
- **Columns**: `model_version`, `mlflow_run_id`, `endpoint_name`

## Version Lifecycle

### Training Phase
```python
# File: src/train_pipeline/pipeline_steps/train.py

# 1. Train model with XGBoost
model = xgb.train(...)

# 2. Log to MLflow
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # 3. Register model (auto-increments version)
    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

    # 4. Get assigned version
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max([int(mv.version) for mv in model_versions])

    # 5. Log version as parameter
    mlflow.log_param("model_version", latest_version)

    # 6. Transition to Staging if quality thresholds met
    if metrics['test_roc_auc'] >= 0.85 and metrics['test_pr_auc'] >= 0.50:
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
```

### Deployment Phase
```python
# File: src/train_pipeline/pipeline_steps/lambda_deploy_endpoint.py

# 1. Extract version from Model Package ARN
version_number = model_package_arn.split('/')[-1]
model_version = f"v{version_number}"

# 2. Set environment variables
current_env['MODEL_VERSION'] = model_version
current_env['MLFLOW_RUN_ID'] = mlflow_run_id

# 3. Create/update endpoint with new environment
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# 4. Validate version after deployment
validation_passed, result = validate_endpoint_version(
    endpoint_name,
    model_version,
    mlflow_run_id
)
```

### Inference Phase
```python
# File: src/train_pipeline/inference_handler.py

# 1. Read version from environment
MODEL_VERSION = os.getenv('MODEL_VERSION', 'unknown')
MLFLOW_RUN_ID = os.getenv('MLFLOW_RUN_ID', 'unknown')

# 2. Include in response
results = {
    "predictions": predictions.tolist(),
    "probabilities": {...},
    "metadata": {
        "model_version": MODEL_VERSION,
        "mlflow_run_id": MLFLOW_RUN_ID,
        "endpoint_name": ENDPOINT_NAME
    }
}
```

## Validation & Testing

### Manual Validation

#### 1. Check MLflow Version
```bash
# Via Python
python -c "
from mlflow.tracking import MlflowClient
from src.config.config import MLFLOW_TRACKING_URI
from src.utils.mlflow_utils import setup_mlflow_tracking

setup_mlflow_tracking(MLFLOW_TRACKING_URI)
client = MlflowClient()
versions = client.search_model_versions(\"name='xgboost-fraud-detector'\")
for v in versions:
    print(f'Version {v.version}: Stage={v.current_stage}, RunID={v.run_id}')
"
```

#### 2. Check SageMaker Endpoint Version
```bash
aws sagemaker describe-endpoint \
    --endpoint-name fraud-detection-endpoint \
    --query 'EndpointConfigName' \
    --output text

# Get model environment variables
aws sagemaker describe-model \
    --model-name <model-name> \
    --query 'PrimaryContainer.Environment'
```

#### 3. Test Inference Response
```bash
python src/train_pipeline/test_endpoint.py \
    --endpoint-name fraud-detection-endpoint \
    --num-samples 1
```

#### 4. Check Athena Logs
```sql
SELECT model_version, mlflow_run_id, COUNT(*) as count
FROM fraud_detection.inference_responses
WHERE endpoint_name = 'fraud-detection-endpoint'
  AND request_timestamp > CURRENT_TIMESTAMP - INTERVAL '1' HOUR
GROUP BY model_version, mlflow_run_id;
```

### Automated Testing

#### Unit Tests
```bash
pytest tests/test_version_validation.py -v
```

#### Integration Test
```python
from src.train_pipeline.test_endpoint import test_version_consistency_end_to_end

# Run version consistency validation
results = test_version_consistency_end_to_end(
    endpoint_name="fraud-detection-endpoint",
    mlflow_model_name="xgboost-fraud-detector"
)

print(f"Status: {results['status']}")
print(f"Validations: {results['validations']}")
print(f"Version Info: {results['version_info']}")
```

## Monitoring & Analytics

### Version Distribution Query
```python
from src.train_pipeline.athena.athena_client import AthenaClient

client = AthenaClient()

# Get version distribution for last 24 hours
dist = client.get_version_distribution(
    endpoint_name="fraud-detection-endpoint",
    hours=24
)
print(dist)
```

### Detect Version Drift
```python
# Check if multiple versions are serving simultaneously
drift_result = client.detect_version_drift(
    endpoint_name="fraud-detection-endpoint",
    hours=1
)

if drift_result['has_drift']:
    print(f"⚠️ Alert: {drift_result['drift_message']}")
    print(f"Versions detected: {drift_result['versions']}")
```

### Version Performance Comparison
```python
# Compare performance across versions
perf = client.get_version_performance_comparison(
    endpoint_name="fraud-detection-endpoint",
    days=7
)

print(perf[['model_version', 'total_predictions', 'fraud_rate', 'avg_latency_ms']])
```

## Troubleshooting

### Issue: Version Mismatch Between Training and Inference

**Symptoms:**
- Inference response shows different version than expected
- Athena logs show unexpected version

**Diagnosis:**
1. Check MLflow latest version:
   ```python
   from mlflow.tracking import MlflowClient
   client = MlflowClient()
   versions = client.search_model_versions("name='xgboost-fraud-detector'")
   print(f"Latest: {max([int(v.version) for v in versions])}")
   ```

2. Check deployed endpoint environment:
   ```bash
   aws sagemaker describe-model --model-name <model> \
       --query 'PrimaryContainer.Environment.MODEL_VERSION'
   ```

3. Test inference response:
   ```bash
   python src/train_pipeline/test_endpoint.py --endpoint-name <endpoint> --num-samples 1
   ```

**Resolution:**
- Redeploy endpoint with correct model version
- Verify MODEL_VERSION environment variable is set correctly
- Run validation test to confirm fix

### Issue: Multiple Versions Serving Simultaneously

**Symptoms:**
- Athena shows multiple versions in recent time window
- Inconsistent predictions for same inputs

**Diagnosis:**
```python
drift_result = client.detect_version_drift("fraud-detection-endpoint", hours=1)
if drift_result['has_drift']:
    print(f"Versions: {drift_result['versions']}")
```

**Resolution:**
- Check if endpoint update is in progress
- Verify no manual changes to endpoint configuration
- If unintended, update endpoint to single version

### Issue: Version Not in Inference Response

**Symptoms:**
- Response missing `metadata` field
- Clients cannot determine model version

**Diagnosis:**
1. Check if inference handler was updated:
   ```bash
   grep -n "metadata" src/train_pipeline/inference_handler.py
   ```

2. Verify environment variables are set:
   ```bash
   aws sagemaker describe-model --model-name <model> \
       --query 'PrimaryContainer.Environment'
   ```

**Resolution:**
- Redeploy inference handler with metadata support
- Ensure MODEL_VERSION environment variable is set
- Test response format

## Best Practices

### 1. Version Naming Convention
- Use semantic format: "v1", "v2", "v3"
- Increment version for every production deployment
- Document breaking changes in MLflow run description

### 2. Version Validation
- Always run validation tests after deployment
- Monitor for version drift alerts
- Set up automated monitoring dashboards

### 3. Version Rollback
```python
# To rollback to previous version:
# 1. Identify previous model package ARN
# 2. Update endpoint with previous ARN
# 3. Verify version in inference response
```

### 4. Version Documentation
- Log version changes in MLflow with detailed notes
- Update team when major version changes
- Document model improvements in version tags

## Monitoring Queries

### Daily Version Report
```sql
SELECT
    DATE(request_timestamp) as date,
    model_version,
    COUNT(*) as predictions,
    AVG(inference_latency_ms) as avg_latency,
    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as fraud_count
FROM fraud_detection.inference_responses
WHERE endpoint_name = 'fraud-detection-endpoint'
  AND request_timestamp > CURRENT_DATE - INTERVAL '7' DAY
GROUP BY DATE(request_timestamp), model_version
ORDER BY date DESC, model_version;
```

### Version Transition Detection
```sql
SELECT
    DATE_TRUNC('hour', request_timestamp) as hour,
    model_version,
    COUNT(*) as count,
    MIN(request_timestamp) as first_seen,
    MAX(request_timestamp) as last_seen
FROM fraud_detection.inference_responses
WHERE endpoint_name = 'fraud-detection-endpoint'
  AND request_timestamp > CURRENT_TIMESTAMP - INTERVAL '24' HOUR
GROUP BY hour, model_version
ORDER BY hour DESC;
```

## Related Files

- **Training**: `src/train_pipeline/pipeline_steps/train.py`
- **Deployment**: `src/train_pipeline/pipeline_steps/lambda_deploy_endpoint.py`
- **Inference**: `src/train_pipeline/inference_handler.py`
- **Testing**: `src/train_pipeline/test_endpoint.py`
- **Analytics**: `src/train_pipeline/athena/athena_client.py`
- **Schema**: `src/train_pipeline/athena/schema_definitions.py`

## Support

For issues or questions about version management:
1. Check this documentation first
2. Review MLflow UI for version history
3. Check CloudWatch logs for deployment errors
4. Run validation tests to diagnose issues
