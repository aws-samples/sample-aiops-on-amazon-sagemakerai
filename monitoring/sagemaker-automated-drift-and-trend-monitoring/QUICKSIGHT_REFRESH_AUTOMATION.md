# QuickSight Dashboard Refresh Automation

## Overview

Added automated refresh functionality to keep the QuickSight governance dashboard up-to-date with the latest drift monitoring results, inference predictions, and ground truth data.

## Changes Made

### 1. Quick Refresh Cell in Governance Dashboard

**File:** `notebooks/3_governance_dashboard.ipynb`

**New Cells Added (at the top):**
- **Cell 1**: Markdown header explaining quick refresh functionality
- **Cell 2**: Code cell to manually refresh datasets with latest Athena data

**What it does:**
- Refreshes all 3 QuickSight datasets in SPICE:
  - Inference Monitoring Dataset
  - Drift Monitoring Dataset
  - Feature Drift Analysis Dataset
- Uses `quicksight.create_ingestion()` API
- Displays refresh status for each dataset
- Checks for unique MLflow run IDs
- Provides helpful error messages if datasets don't exist

**Usage:**
```python
# Just run Cell 2 in the governance dashboard notebook
# Takes 30-60 seconds to refresh all datasets
```

**When to use:**
- After new drift monitoring runs complete
- After adding new inference predictions
- After updating ground truth
- When you want to see the latest data immediately

### 2. Automated Daily Refresh in Inference Monitoring Notebooks

**Files:**
- `notebooks/2a_inference_monitoring.ipynb` (Section 7.9)
- `notebooks/2b_inference_monitoring_WIP.ipynb` (Section 13)

**New Cells Added (at the end, before summary):**
1. **Markdown header**: Explains automated refresh setup
2. **Lambda function creation**: Creates `quicksight-dashboard-refresh` Lambda
3. **EventBridge rule**: Schedules daily refresh at 3:00 AM UTC

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Automated Refresh Flow                       │
└─────────────────────────────────────────────────────────────────┘

2:00 AM UTC ─────► Drift Monitoring Lambda runs (EventBridge)
                   │
                   ├─► Analyzes data drift (Evidently)
                   ├─► Analyzes model drift (Evidently)
                   ├─► Logs to MLflow (with unique run ID)
                   └─► Writes to Athena monitoring_responses table

                   ⏱️  Wait 1 hour...

3:00 AM UTC ─────► QuickSight Refresh Lambda runs (EventBridge)
                   │
                   ├─► Triggers SPICE ingestion for 3 datasets:
                   │   ├─ Inference Monitoring Dataset
                   │   ├─ Drift Monitoring Dataset
                   │   └─ Feature Drift Analysis Dataset
                   │
                   └─► QuickSight dashboard shows updated data

Morning ─────────► Users see latest drift analysis in dashboard! 📊
```

## Components Created

### Lambda Function: `quicksight-dashboard-refresh`

**Runtime:** Python 3.11
**Timeout:** 60 seconds
**Memory:** 128 MB
**Trigger:** EventBridge scheduled rule (3:00 AM UTC daily)

**Environment Variables:**
- `AWS_ACCOUNT_ID`: AWS account ID
- `INFERENCE_DATASET_ID`: QuickSight inference dataset ID
- `DRIFT_DATASET_ID`: QuickSight drift dataset ID
- `FEATURE_DRIFT_DATASET_ID`: QuickSight feature drift dataset ID

**Permissions:**
- `quicksight:CreateIngestion` - Trigger dataset refresh
- `quicksight:DescribeIngestion` - Check refresh status
- `quicksight:ListIngestions` - List ingestions

**What it does:**
1. Connects to QuickSight
2. Iterates through 3 dataset IDs
3. Creates a new ingestion (refresh) for each dataset
4. Returns success/failure for each dataset

### IAM Role: `quicksight-dashboard-refresh-role`

**Trust Policy:** Lambda service can assume this role

**Managed Policies:**
- `AWSLambdaBasicExecutionRole` - CloudWatch Logs access

**Inline Policies:**
- `QuickSightDatasetRefresh` - QuickSight ingestion permissions

### EventBridge Rule: `quicksight-dashboard-daily-refresh`

**Schedule:** `cron(0 3 * * ? *)` (3:00 AM UTC daily)
**State:** ENABLED
**Target:** `quicksight-dashboard-refresh` Lambda
**Description:** Daily QuickSight dataset refresh at 3 AM UTC (after drift monitoring)

## Schedule Summary

| Time (UTC) | Event | Description |
|------------|-------|-------------|
| **2:00 AM** | Drift Monitoring Lambda | Runs Evidently drift detection, logs to MLflow, writes to Athena |
| **3:00 AM** | QuickSight Refresh Lambda | Refreshes SPICE datasets from Athena tables |
| **Morning** | Dashboard Updated | Users see latest drift metrics, inferences, ground truth |

**Why 1 hour delay?**
- Ensures all drift monitoring data is written to Athena
- Allows SQS→Lambda→Athena pipeline to complete
- Prevents race conditions with incomplete data

## Usage

### Initial Setup

Run the new cells at the end of either notebook:

**Option A: 2a_inference_monitoring.ipynb**
```python
# Run these cells in order:
# Cell 7.9: Section header
# Cell: Lambda function creation (creates/updates Lambda)
# Cell: EventBridge rule creation (schedules daily refresh)
```

**Option B: 2b_inference_monitoring_WIP.ipynb**
```python
# Run these cells in order:
# Cell 13: Section header
# Cell: Lambda function creation
# Cell: EventBridge rule creation
```

### Manual Refresh

#### From Notebook (Quick Refresh)
```python
# In 3_governance_dashboard.ipynb, Cell 2
# Just run the cell - takes 30-60 seconds
```

#### From AWS CLI
```bash
aws lambda invoke \
  --function-name quicksight-dashboard-refresh \
  --region us-east-1 \
  output.json

cat output.json
```

#### From AWS Console
1. Go to Lambda console: https://console.aws.amazon.com/lambda
2. Find function: `quicksight-dashboard-refresh`
3. Click "Test" tab
4. Create test event (empty JSON: `{}`)
5. Click "Test" button

### Verify Schedule

**Check EventBridge Rule:**
```bash
aws events describe-rule \
  --name quicksight-dashboard-daily-refresh \
  --region us-east-1
```

**Expected Output:**
```json
{
  "Name": "quicksight-dashboard-daily-refresh",
  "Arn": "arn:aws:events:us-east-1:...",
  "ScheduleExpression": "cron(0 3 * * ? *)",
  "State": "ENABLED",
  "Description": "Daily QuickSight dataset refresh at 3 AM UTC (after drift monitoring)"
}
```

**Check Lambda Permissions:**
```bash
aws lambda get-policy \
  --function-name quicksight-dashboard-refresh \
  --region us-east-1
```

Should show EventBridge has permission to invoke the Lambda.

### Monitor Execution

**CloudWatch Logs:**
```bash
aws logs tail /aws/lambda/quicksight-dashboard-refresh --follow
```

**Expected Log Output:**
```
✓ Refreshed dataset: fraud-governance-inference-dataset
✓ Refreshed dataset: fraud-governance-drift-dataset
✓ Refreshed dataset: fraud-governance-feature-drift-dataset
```

**QuickSight Console:**
1. Go to: https://quicksight.aws.amazon.com
2. Click "Datasets" in left menu
3. Find your datasets
4. Click "Refresh" tab to see ingestion history
5. Should see daily ingestions at 3:00 AM UTC

### Troubleshooting

#### Dataset Not Found Error
```
ResourceNotFoundException: Dataset 'fraud-governance-...' not found
```

**Solution:** Create the dashboard first using `3_governance_dashboard.ipynb`

#### Permission Denied Error
```
AccessDeniedException: User is not authorized to perform quicksight:CreateIngestion
```

**Solution:** Check IAM role has QuickSight permissions:
```bash
aws iam get-role-policy \
  --role-name quicksight-dashboard-refresh-role \
  --policy-name QuickSightDatasetRefresh
```

#### EventBridge Not Triggering
```bash
# Check rule is enabled
aws events describe-rule --name quicksight-dashboard-daily-refresh

# Check targets are configured
aws events list-targets-by-rule --rule quicksight-dashboard-daily-refresh
```

#### Ingestion Failed in QuickSight

**Check ingestion status:**
```bash
aws quicksight list-ingestions \
  --aws-account-id <ACCOUNT_ID> \
  --data-set-id <DATASET_ID>
```

**Common causes:**
- Athena tables empty (no data to refresh)
- SPICE capacity limit reached
- Dataset permissions issue
- Athena query timeout

### Disable Automated Refresh

If you want to stop the daily refresh:

```python
# In the notebook
import boto3
events = boto3.client('events', region_name='us-east-1')

events.disable_rule(Name='quicksight-dashboard-daily-refresh')
print("✓ Automated refresh disabled")
```

Or via AWS CLI:
```bash
aws events disable-rule \
  --name quicksight-dashboard-daily-refresh \
  --region us-east-1
```

To re-enable:
```bash
aws events enable-rule \
  --name quicksight-dashboard-daily-refresh \
  --region us-east-1
```

### Delete Resources

If you want to remove the automation:

```python
import boto3

lambda_client = boto3.client('lambda', region_name='us-east-1')
events = boto3.client('events', region_name='us-east-1')
iam = boto3.client('iam', region_name='us-east-1')

# Remove EventBridge rule
events.remove_targets(Rule='quicksight-dashboard-daily-refresh', Ids=['1'])
events.delete_rule(Name='quicksight-dashboard-daily-refresh')
print("✓ Deleted EventBridge rule")

# Delete Lambda function
lambda_client.delete_function(FunctionName='quicksight-dashboard-refresh')
print("✓ Deleted Lambda function")

# Delete IAM role (detach policies first)
iam.detach_role_policy(
    RoleName='quicksight-dashboard-refresh-role',
    PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
)
iam.delete_role_policy(
    RoleName='quicksight-dashboard-refresh-role',
    PolicyName='QuickSightDatasetRefresh'
)
iam.delete_role(RoleName='quicksight-dashboard-refresh-role')
print("✓ Deleted IAM role")
```

## Benefits

1. **Always Up-to-Date**: Dashboard automatically shows latest data every morning
2. **No Manual Work**: Set it up once, runs forever
3. **Proper Timing**: Refreshes 1 hour after drift monitoring completes
4. **Cost Efficient**: Only refreshes once per day, not continuously
5. **Flexible**: Can manually refresh anytime with quick refresh cell
6. **Reliable**: EventBridge ensures execution even if notebook not running

## Cost Considerations

**Lambda:**
- Invocations: 1 per day = ~30/month
- Duration: ~5 seconds per invocation
- Cost: **~$0.00** (well within free tier)

**QuickSight:**
- SPICE ingestions: 3 datasets × 1/day = 90/month
- SPICE storage: Depends on data size
- Cost: **Minimal** (SPICE refreshes usually included in QuickSight subscription)

**EventBridge:**
- Rule invocations: 1 per day = ~30/month
- Cost: **$0.00** (first 1M invocations/month free)

**Total Monthly Cost:** < $1 (essentially free)

## Related Documentation

- MLflow Run ID Fix: `MLFLOW_RUN_ID_FIX.md`
- Governance Dashboard Setup: `notebooks/3_governance_dashboard.ipynb`
- Drift Monitoring Setup: `notebooks/2a_inference_monitoring.ipynb`
- AWS QuickSight API: https://docs.aws.amazon.com/quicksight/latest/APIReference/
- AWS EventBridge Cron: https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-create-rule-schedule.html
