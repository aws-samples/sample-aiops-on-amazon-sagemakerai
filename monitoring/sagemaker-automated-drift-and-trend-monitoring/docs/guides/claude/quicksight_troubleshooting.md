# QuickSight Dashboard Troubleshooting Guide

## Error: "Unknown parameter PrimaryYAxisDisplayOptions: AxisLineVisibility"

### Root Cause
The Python variables `DRIFT_VISUALS` and `FEATURE_DRIFT_VISUALS` in your Jupyter kernel memory have old structure from before the fixes were applied.

### Solution
After restarting the kernel, you MUST re-execute these cells in order:

```
Cell 2  → Import config variables
Cell 12 → Create inference dataset (sets DATASET_ARN)
Cell 14 → Create drift dataset (sets DRIFT_DATASET_ARN)
Cell 16 → Create feature drift dataset (sets FEATURE_DRIFT_DATASET_ARN)
Cell 18 → Define INFERENCE_VISUALS
Cell 20 → Define DRIFT_VISUALS ⚠️ MUST RUN
Cell 22 → Define FEATURE_DRIFT_VISUALS ⚠️ MUST RUN
Cell 24 → Create/update analysis
```

### Verification Cell

Add this cell **after Cell 22** and **before Cell 24** to verify your variables are correct:

```python
# VERIFICATION: Check visual structure before creating analysis
import json

def verify_visuals():
    """Verify DRIFT_VISUALS and FEATURE_DRIFT_VISUALS have correct structure"""
    issues = []

    # Check DRIFT_VISUALS
    for i, visual in enumerate(DRIFT_VISUALS):
        if 'LineChartVisual' in visual:
            config = visual['LineChartVisual']['ChartConfiguration']
            if 'PrimaryYAxisDisplayOptions' in config:
                opts = config['PrimaryYAxisDisplayOptions']
                if 'AxisLineVisibility' in opts:
                    issues.append(f"DRIFT_VISUALS[{i}]: AxisLineVisibility at top level (needs AxisOptions wrapper)")
                elif 'AxisOptions' not in opts:
                    issues.append(f"DRIFT_VISUALS[{i}]: Missing AxisOptions wrapper")

    # Check FEATURE_DRIFT_VISUALS
    for i, visual in enumerate(FEATURE_DRIFT_VISUALS):
        if 'LineChartVisual' in visual:
            config = visual['LineChartVisual']['ChartConfiguration']
            if 'PrimaryYAxisDisplayOptions' in config:
                opts = config['PrimaryYAxisDisplayOptions']
                if 'AxisLineVisibility' in opts:
                    issues.append(f"FEATURE_DRIFT_VISUALS[{i}]: AxisLineVisibility at top level")
                elif 'AxisOptions' not in opts:
                    issues.append(f"FEATURE_DRIFT_VISUALS[{i}]: Missing AxisOptions wrapper")

        # Check DateMeasureField
        if 'TableVisual' in visual:
            fw = visual['TableVisual']['ChartConfiguration']['FieldWells']['TableAggregatedFieldWells']
            for val in fw.get('Values', []):
                if 'DateMeasureField' in val:
                    agg = val['DateMeasureField']['AggregationFunction']
                    if isinstance(agg, dict):
                        issues.append(f"FEATURE_DRIFT_VISUALS[{i}]: DateMeasureField AggregationFunction is dict (should be string)")

    if issues:
        print("❌ VALIDATION FAILED - Variables have old structure:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  You MUST re-run cells 20 and 22 to fix this!")
        print("    The notebook file is correct, but your Python variables are outdated.")
        return False
    else:
        print("✅ VALIDATION PASSED")
        print(f"  - DRIFT_VISUALS: {len(DRIFT_VISUALS)} visuals with correct structure")
        print(f"  - FEATURE_DRIFT_VISUALS: {len(FEATURE_DRIFT_VISUALS)} visuals with correct structure")
        print("\n✓ Safe to proceed with analysis creation (Cell 24)")
        return True

verify_visuals()
```

### If Verification Fails

1. **Re-run Cell 20** (defines DRIFT_VISUALS with corrected structure)
2. **Re-run Cell 22** (defines FEATURE_DRIFT_VISUALS with corrected structure)
3. **Re-run the verification cell** - should now show ✅ VALIDATION PASSED
4. **Run Cell 24** to create/update the analysis

### Quick Fix: Restart & Run All

The safest approach:
1. **Kernel → Restart & Run All**
2. Wait for all cells to complete
3. The analysis will be created with the correct structure

## Current Status

- ✅ All 3 datasets exist and are up-to-date
  - `fraud-governance-inference-dataset`
  - `fraud-governance-drift-dataset`
  - `fraud-governance-feature-drift-dataset`
- ✅ Notebook file has correct code
- ⚠️ Analysis is in `UPDATE_FAILED` status (from previous attempt with wrong structure)

Once you run the corrected cells, the analysis will update successfully.

## Error: "COLUMN_NOT_FOUND: Column 'm.drift_severity' cannot be resolved"

### Root Cause
The Custom SQL query in Cell 16 (feature drift dataset) references `drift_severity` column which doesn't exist in the `monitoring_responses` table.

### Available Columns
The actual columns in `monitoring_responses` table:
- `data_drift_detected` (boolean)
- `drifted_columns_share` (decimal - percentage of drifted features)
- `drifted_columns_count` (integer)
- `roc_auc_degradation_pct` (decimal - model performance degradation)
- ❌ NO `drift_severity` column

### Solution
The query has been fixed to remove the non-existent `drift_severity` column:

**Changes:**
1. Removed `m.drift_severity` from SELECT clause
2. Updated `GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14` → `GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13`
3. Removed `{'Name': 'drift_severity', 'Type': 'STRING'}` from Columns definition

**To Apply Fix:**
1. **Re-run Cell 16** to update the feature drift dataset with corrected SQL
2. The dataset will be recreated without the non-existent column
3. Cell 22 and 24 should then work without errors

### Verification
After re-running Cell 16, check the output:
```
✓ Feature drift dataset: arn:aws:quicksight:...
```

The dataset should now query successfully without COLUMN_NOT_FOUND errors.

## Error: "COLUMN_TYPE_INCOMPATIBLE: NumericalMeasureField can only refer to INTEGER, DECIMAL"

### Root Cause
QuickSight's `NumericalMeasureField` requires numeric column types (INTEGER or DECIMAL), even when using COUNT aggregation. String columns like `inference_id` and `monitoring_run_id` cannot be used.

### Errors Seen
```
COLUMN_TYPE_INCOMPATIBLE: Object NumericalMeasureField can only refer to columns of types [INTEGER, DECIMAL], but the column inference_id is of type STRING.
COLUMN_TYPE_INCOMPATIBLE: Object NumericalMeasureField can only refer to columns of types [INTEGER, DECIMAL], but the column monitoring_run_id is of type STRING.
```

### Solution
Changed to use numeric columns for COUNT operations:

**Cell 18 (INFERENCE_VISUALS)**:
- Before: `col('inference_id')` with COUNT - ❌ STRING column
- After: `col('probability_fraud')` with COUNT - ✅ DECIMAL column

**Cell 20 (DRIFT_VISUALS)**:
- Before: `dcol('monitoring_run_id')` with COUNT - ❌ STRING column
- After: `dcol('drifted_columns_count')` with COUNT - ✅ INTEGER column

**Cell 22 (FEATURE_DRIFT_VISUALS)**:
- Before: `fcol('monitoring_run_id')` with COUNT - ❌ STRING column
- After: `fcol('drifted_columns_count')` with COUNT - ✅ INTEGER column

### Why This Works
- COUNT aggregation counts non-null rows regardless of column
- Using a numeric column (probability_fraud, drifted_columns_count) gives the same count result
- QuickSight accepts NumericalMeasureField with numeric column types

### To Apply Fix
1. **Re-run Cell 18** to reload INFERENCE_VISUALS with numeric columns
2. **Re-run Cell 20** to reload DRIFT_VISUALS with numeric columns
3. **Re-run Cell 22** to reload FEATURE_DRIFT_VISUALS with numeric columns
4. **Re-run Cell 23** to verify (should show ✅ PASSED)
5. **Re-run Cell 25** to update analysis (should succeed now)
6. **Re-run Cell 27** to update dashboard
