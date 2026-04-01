# MLflow Run ID Fix for Drift Monitoring

## Problem
The drift monitoring Lambda was creating new MLflow runs for each drift test, but the run IDs were not being captured and stored in the `monitoring_responses` Athena table. This made it impossible to trace which drift test results came from which MLflow run.

## Root Cause
In `src/drift_monitoring/lambda_drift_monitor.py`:

1. **`log_to_mlflow()` function** (line 638):
   - Created a new MLflow run with `mlflow.start_run()`
   - Logged metrics and artifacts to MLflow
   - But did NOT return the run ID

2. **`lambda_handler()` function** (line 827):
   - Called `log_to_mlflow()` without capturing the return value
   - Called `write_monitoring_results()` without passing the run ID
   - Result: `mlflow_run_id` column in `monitoring_responses` was always NULL

## Solution

### Changes Made

**File:** `src/drift_monitoring/lambda_drift_monitor.py`

#### 1. Modified `log_to_mlflow()` to return run ID (lines 638-750)

**Before:**
```python
def log_to_mlflow(data_drift_result, model_drift_result):
    """Log drift metrics and Evidently HTML reports to MLflow."""
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow not available - skipping MLflow logging")
        return

    # ...
    with mlflow.start_run(run_name=f"drift-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Log metrics...
        print("✓ Successfully logged Evidently reports and metrics to MLflow")

    except Exception as e:
        print(f"⚠️ Failed to log to MLflow: {e}")
```

**After:**
```python
def log_to_mlflow(data_drift_result, model_drift_result):
    """Log drift metrics and Evidently HTML reports to MLflow.

    Returns:
        str: The MLflow run ID, or None if logging failed/skipped
    """
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow not available - skipping MLflow logging")
        return None

    # ...
    with mlflow.start_run(run_name=f"drift-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        # Capture the run ID
        run_id = run.info.run_id

        # Log metrics...
        print("✓ Successfully logged Evidently reports and metrics to MLflow")
        print(f"  MLflow Run ID: {run_id}")

        return run_id

    except Exception as e:
        print(f"⚠️ Failed to log to MLflow: {e}")
        return None
```

#### 2. Modified `lambda_handler()` to capture and pass run ID (lines 831-849)

**Before:**
```python
# Log Evidently reports and metrics to MLflow
log_to_mlflow(data_drift_result, model_drift_result)

# Send alert if drift detected
send_sns_alert(data_drift_result, model_drift_result)

# Write monitoring results to SQS → Athena
write_monitoring_results(data_drift_result, model_drift_result)
```

**After:**
```python
# Log Evidently reports and metrics to MLflow (captures run ID)
mlflow_run_id = log_to_mlflow(data_drift_result, model_drift_result)

# Send alert if drift detected
send_sns_alert(data_drift_result, model_drift_result)

# Write monitoring results to SQS → Athena (with MLflow run ID)
write_monitoring_results(data_drift_result, model_drift_result, mlflow_run_id)
```

## Verification

### 1. Check Drift Monitoring Runs in MLflow UI

Navigate to your MLflow tracking server and verify:
- Experiment: `fraud-detection-drift_monitoring`
- Each drift test creates a new run with name format: `drift-check-YYYYMMDD-HHMMSS`
- Run IDs are unique UUIDs (e.g., `a1b2c3d4e5f6...`)

### 2. Query monitoring_responses Table

Add this cell to `notebooks/4_optional_version_validation.ipynb`:

```python
# Query monitoring_responses for drift test runs
query = f"""
    SELECT
        monitoring_run_id,
        monitoring_timestamp,
        mlflow_run_id,
        data_drift_detected,
        model_drift_detected,
        drifted_columns_count,
        current_roc_auc,
        detection_engine
    FROM {ATHENA_DATABASE}.monitoring_responses
    ORDER BY monitoring_timestamp DESC
    LIMIT 10
"""

athena_df = athena_client.run_query(query)

print("\n" + "=" * 80)
print("Drift Monitoring Runs (Most Recent 10)")
print("=" * 80)

if not athena_df.empty:
    print(athena_df.to_string(index=False))
    print(f"\n✓ Total runs: {len(athena_df)}")

    # Check for unique MLflow run IDs
    unique_runs = athena_df['mlflow_run_id'].nunique()
    print(f"✓ Unique MLflow run IDs: {unique_runs}")

    if unique_runs == len(athena_df):
        print("✅ Each drift test has a unique MLflow run ID")
    else:
        print("⚠️  Some drift tests share MLflow run IDs (check for issues)")
else:
    print("No drift monitoring runs found")
```

**Expected Output:**
```
================================================================================
Drift Monitoring Runs (Most Recent 10)
================================================================================
monitoring_run_id    monitoring_timestamp     mlflow_run_id                    data_drift_detected  model_drift_detected  ...
drift-20260324-...   2026-03-24 15:30:00      a1b2c3d4e5f6789...              True                 False                 ...
drift-20260324-...   2026-03-24 14:00:00      f6e5d4c3b2a1987...              False                False                 ...
drift-20260323-...   2026-03-23 14:00:00      9876543210abcde...              False                True                  ...

✓ Total runs: 10
✓ Unique MLflow run IDs: 10
✅ Each drift test has a unique MLflow run ID
```

### 3. Verify Run ID Consistency

Check that the MLflow run IDs in `monitoring_responses` match the actual runs in MLflow:

```python
# Get most recent drift monitoring run
query = f"""
    SELECT
        mlflow_run_id,
        monitoring_timestamp,
        data_drift_detected,
        model_drift_detected
    FROM {ATHENA_DATABASE}.monitoring_responses
    ORDER BY monitoring_timestamp DESC
    LIMIT 1
"""

athena_df = athena_client.run_query(query)

if not athena_df.empty:
    athena_run_id = athena_df['mlflow_run_id'].iloc[0]
    print(f"Most recent Athena run ID: {athena_run_id}")

    # Query MLflow for this run
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        run = mlflow.get_run(athena_run_id)
        print(f"✅ MLflow run found: {run.info.run_name}")
        print(f"   Start time: {run.info.start_time}")
        print(f"   Status: {run.info.status}")
        print(f"   Metrics: {list(run.data.metrics.keys())[:5]}...")
    except Exception as e:
        print(f"❌ MLflow run not found: {e}")
```

### 4. Test After Lambda Deployment

After deploying the updated drift monitoring Lambda:

1. **Trigger a manual drift test:**
   ```python
   # In notebook 2a_inference_monitoring.ipynb
   lambda_client = boto3.client('lambda', region_name='us-east-1')

   response = lambda_client.invoke(
       FunctionName='fraud-detection-drift-monitor',
       InvocationType='RequestResponse'
   )

   result = json.loads(response['Payload'].read())
   print(json.dumps(result, indent=2))
   ```

2. **Wait ~30 seconds for SQS processing**

3. **Query monitoring_responses:**
   ```sql
   SELECT
       mlflow_run_id,
       monitoring_timestamp,
       data_drift_detected,
       model_drift_detected
   FROM fraud_detection.monitoring_responses
   ORDER BY monitoring_timestamp DESC
   LIMIT 1
   ```

4. **Verify the run ID is NOT NULL and is a valid UUID**

## Deployment

To deploy the fix:

1. **Update the drift monitoring Lambda:**
   ```bash
   cd /path/to/monitoring/sagemaker-automated-drift-and-trend-monitoring
   bash scripts/deploy_lambda_container.sh your-email@example.com 0.2 0.05
   ```

   Or if you already have the Lambda deployed, force redeployment:
   ```python
   # In notebook 2a_inference_monitoring.ipynb
   REDEPLOY_LAMBDAS = True  # Force redeployment

   # Then run Cell 59 (drift monitor deployment)
   ```

2. **Test the fix:**
   - Run drift monitoring (Cell 62 in notebook)
   - Check CloudWatch logs for "MLflow Run ID: ..." message
   - Query `monitoring_responses` table
   - Verify `mlflow_run_id` is populated with unique UUIDs

## Benefits

1. **Traceability**: Every drift test now has a unique MLflow run ID
2. **Audit Trail**: Can track which drift detection results came from which MLflow run
3. **Debugging**: Easy to correlate Athena records with MLflow artifacts (HTML reports, metrics)
4. **Governance**: Version tracking works correctly across all tables

## Notes

- The `inference_responses` table's `mlflow_run_id` column is DIFFERENT:
  - It stores the run ID from when the MODEL was deployed (set in endpoint environment variable)
  - Typically shows "unknown" or "pipeline" for endpoints deployed outside of notebooks
  - This is CORRECT behavior - it tracks the model version, not the drift test version

- The `monitoring_responses` table's `mlflow_run_id` column:
  - Stores the run ID from the DRIFT TEST itself
  - Should be unique for each drift monitoring run
  - Allows tracking of drift detection over time

## Related Files

- `src/drift_monitoring/lambda_drift_monitor.py` - Main drift monitoring Lambda (UPDATED)
- `src/drift_monitoring/lambda_monitoring_writer.py` - Writes to monitoring_responses table
- `notebooks/4_optional_version_validation.ipynb` - Version validation queries
- `notebooks/2a_inference_monitoring.ipynb` - Manual drift testing
