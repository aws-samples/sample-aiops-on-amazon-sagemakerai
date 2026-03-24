# QuickSight Dashboard/Analysis Cleanup

## When Dashboard/Analysis is in FAILED State

If you see errors like:
- `ConflictException: Cannot publish a dashboard version when it is in CREATION_FAILED state`
- `Analysis in UPDATE_FAILED state`

This means a previous update attempt failed and left the resource in a bad state.

## Solution 1: Let Auto-Recovery Work (Recommended)

The notebook cells (25 for analysis, 27 for dashboard) now have **automatic status checking**:

1. After update, they wait for the resource to reach `CREATION_SUCCESSFUL` state
2. If it reaches `CREATION_FAILED` or `UPDATE_FAILED`, they show the error
3. You can fix the underlying issue and re-run the cell

**To recover:**
1. Fix all structural issues (already done)
2. Re-run the cells in order:
   - Cell 16: Update feature drift dataset (fixed SQL)
   - Cell 20: Define DRIFT_VISUALS (fixed structure)
   - Cell 22: Define FEATURE_DRIFT_VISUALS (fixed structure)
   - Cell 23: Verify visuals
   - Cell 25: Update analysis (will wait for success)
   - Cell 27: Update dashboard (will wait for success)

## Solution 2: Delete and Recreate

If the auto-recovery doesn't work, manually delete the failed resources:

### Delete Failed Analysis

```python
import boto3
from src.config.config import QUICKSIGHT_ANALYSIS_ID, AWS_DEFAULT_REGION

sts = boto3.client('sts', region_name=AWS_DEFAULT_REGION)
account_id = sts.get_caller_identity()['Account']

quicksight = boto3.client('quicksight', region_name=AWS_DEFAULT_REGION)

try:
    quicksight.delete_analysis(
        AwsAccountId=account_id,
        AnalysisId=QUICKSIGHT_ANALYSIS_ID,
        ForceDeleteWithoutRecovery=True
    )
    print(f'✓ Deleted analysis: {QUICKSIGHT_ANALYSIS_ID}')
except Exception as e:
    print(f'Error: {e}')
```

### Delete Failed Dashboard

```python
import boto3
from src.config.config import QUICKSIGHT_DASHBOARD_ID, AWS_DEFAULT_REGION

sts = boto3.client('sts', region_name=AWS_DEFAULT_REGION)
account_id = sts.get_caller_identity()['Account']

quicksight = boto3.client('quicksight', region_name=AWS_DEFAULT_REGION)

try:
    quicksight.delete_dashboard(
        AwsAccountId=account_id,
        DashboardId=QUICKSIGHT_DASHBOARD_ID
    )
    print(f'✓ Deleted dashboard: {QUICKSIGHT_DASHBOARD_ID}')
except Exception as e:
    print(f'Error: {e}')
```

### After Deletion

Re-run cells 25 and 27. They will detect the resources don't exist and create them fresh.

## Check Current Status

Use this to see the current state of your resources:

```python
import boto3
from src.config.config import (
    QUICKSIGHT_ANALYSIS_ID,
    QUICKSIGHT_DASHBOARD_ID,
    AWS_DEFAULT_REGION
)

sts = boto3.client('sts', region_name=AWS_DEFAULT_REGION)
account_id = sts.get_caller_identity()['Account']

quicksight = boto3.client('quicksight', region_name=AWS_DEFAULT_REGION)

# Check Analysis
try:
    resp = quicksight.describe_analysis(
        AwsAccountId=account_id,
        AnalysisId=QUICKSIGHT_ANALYSIS_ID
    )
    print(f"Analysis Status: {resp['Analysis']['Status']}")
    if resp['Analysis']['Status'] in ['CREATION_FAILED', 'UPDATE_FAILED']:
        print("  ⚠️  Analysis is in failed state")
except Exception as e:
    print(f"Analysis: NOT FOUND or ERROR")

# Check Dashboard
try:
    resp = quicksight.describe_dashboard(
        AwsAccountId=account_id,
        DashboardId=QUICKSIGHT_DASHBOARD_ID
    )
    print(f"Dashboard Status: {resp['Dashboard']['Version']['Status']}")
    if resp['Dashboard']['Version']['Status'] in ['CREATION_FAILED', 'UPDATE_FAILED']:
        print("  ⚠️  Dashboard is in failed state")
except Exception as e:
    print(f"Dashboard: NOT FOUND or ERROR")
```

## Prevention

The updated cells now have built-in status checking to prevent leaving resources in failed states:

- ✅ **Cell 25**: Waits for analysis to succeed before continuing
- ✅ **Cell 27**: Waits for dashboard to succeed before publishing
- ✅ **Cell 23**: Validates visual structure before creating analysis

Run Cell 23 before Cell 25 to catch structural issues early.
