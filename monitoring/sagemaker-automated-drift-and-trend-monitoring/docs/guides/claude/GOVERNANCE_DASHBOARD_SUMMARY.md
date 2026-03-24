# QuickSight Governance Dashboard - Feature Drift Analysis

## Overview

The QuickSight governance dashboard now includes **3 sheets with 16 visuals** for comprehensive inference and drift monitoring. The latest addition is **Sheet 3: Feature Drift Analysis** which shows drift trends across multiple model versions.

## Dashboard Structure

### Sheet 1: Inference Monitoring (6 visuals)
- **Purpose:** Monitor real-time prediction behavior
- **Data Source:** `inference_responses` table
- **Visuals:**
  1. Prediction Volume Over Time
  2. Fraud Probability Distribution
  3. Prediction Accuracy Breakdown
  4. Risk Tier Distribution
  5. Inference Latency Trend
  6. Total Inferences KPI

### Sheet 2: Drift Trend Analysis (6 visuals)
- **Purpose:** Track data and model drift metrics over time
- **Data Source:** `monitoring_responses` table
- **Visuals:**
  7. Data Drift Share Over Time
  8. Drifted Features Count Trend
  9. ROC-AUC Degradation Trend (baseline vs current)
  10. Model Performance Metrics (accuracy, precision, recall, F1)
  11. Drift Alerts Timeline
  12. Latest Drift Share KPI

### Sheet 3: Feature Drift Analysis (4 visuals) ⭐ NEW
- **Purpose:** Analyze drift patterns by model version and correlate with inference volume
- **Data Source:** Custom SQL joining `inference_responses` + `monitoring_responses`
- **Visuals:**
  13. **Drift by Model Version Over Time** (multi-line chart)
      - Shows how drift evolves for each model version
      - Enables comparison of model stability
  14. **Model Version Performance Summary** (table)
      - Latest drift share per version
      - Average ROC-AUC per version
      - Number of monitoring runs
      - Last check timestamp
  15. **Inference Volume vs Drift Correlation** (combo chart)
      - Bars: Daily inference count
      - Line: Drift percentage
      - Shows if high volume correlates with drift
  16. **Drift Intensity Heatmap** (pivot table)
      - Rows: Time (by day)
      - Columns: Model versions
      - Values: Drift share (color-coded)
      - Quickly identify when/where drift occurs

## Key Features

### Joined Dataset
The feature drift dataset uses Custom SQL to create a comprehensive view:

```sql
SELECT
    m.monitoring_run_id,
    m.monitoring_timestamp,
    m.model_version,
    m.drifted_columns_count,
    m.drifted_columns_share,
    m.baseline_roc_auc,
    m.current_roc_auc,
    m.drift_severity,
    COUNT(DISTINCT i.inference_id) as inference_count,
    AVG(i.probability_fraud) as avg_fraud_prob
FROM monitoring_responses m
LEFT JOIN inference_responses i
    ON i.model_version = m.model_version
    AND i.request_timestamp >= m.monitoring_timestamp - INTERVAL '1' DAY
GROUP BY ...
```

**Benefits:**
- ✅ Correlates drift with inference activity
- ✅ Tracks drift across model version deployments
- ✅ Shows if drift increases with prediction volume
- ✅ Enables A/B testing comparison between models

### Configuration

All dataset IDs and names are configurable via `config.yaml`:

```yaml
quicksight:
  feature_drift_dataset_id: fraud-governance-feature-drift-dataset
  feature_drift_dataset_name: Fraud Governance - Feature Drift Analysis
```

Or override with environment variables:
```bash
export QUICKSIGHT_FEATURE_DRIFT_DATASET_ID=my-custom-id
export QUICKSIGHT_FEATURE_DRIFT_DATASET_NAME="My Dashboard"
```

## Use Cases

### 1. Model Rollout Monitoring
**Scenario:** You deployed a new model version (v2.1)

**How to use:**
- Open Sheet 3, Visual 13 (Drift by Model Version)
- Compare drift lines for v2.0 vs v2.1
- Check if new version has higher/lower drift
- Review Visual 14 table for side-by-side comparison

### 2. Drift Root Cause Analysis
**Scenario:** Drift spike detected, need to investigate cause

**How to use:**
- Sheet 3, Visual 15 (Volume vs Drift)
- Check if drift spike correlates with volume spike
- If yes → likely data distribution shift
- If no → investigate model degradation or feature engineering changes

### 3. Production Model Health
**Scenario:** Weekly review of all production models

**How to use:**
- Sheet 3, Visual 14 (Model Version Summary)
- Sort by "Last Check" to see recent activity
- Check "Latest Drift Share" column for current health
- Review "Avg ROC-AUC" for model quality

### 4. Time-Series Drift Patterns
**Scenario:** Looking for temporal drift patterns

**How to use:**
- Sheet 3, Visual 16 (Heatmap)
- Rows = days, Columns = model versions
- Color intensity shows drift level
- Quickly spot: weekday vs weekend patterns, monthly cycles, gradual vs sudden drift

## Limitations & Future Enhancements

### Current Limitations

**1. Feature-Level Granularity**
- The `per_feature_drift_scores` column is stored as JSON
- QuickSight Custom SQL has limited JSON parsing capabilities
- Visuals show aggregate drift, not individual feature scores

**2. Model Version Tracking**
- Assumes `model_version` is consistently populated in both tables
- If version changes mid-day, the 1-day join window may include mixed data

### Recommended Enhancements

**Option A: Create Athena View for Per-Feature Analysis**
```sql
CREATE OR REPLACE VIEW feature_drift_detail AS
SELECT
    monitoring_run_id,
    monitoring_timestamp,
    model_version,
    feature_name,
    CAST(json_extract(feature_data, '$.drift_score') AS DOUBLE) as drift_score,
    CAST(json_extract(feature_data, '$.drift_detected') AS BOOLEAN) as drift_detected
FROM monitoring_responses
CROSS JOIN UNNEST(
    CAST(json_parse(per_feature_drift_scores) AS MAP(VARCHAR, VARCHAR))
) AS t(feature_name, feature_data);
```

**Benefits:**
- ✅ Creates one row per feature per monitoring run
- ✅ Easy to visualize feature-level drift in QuickSight
- ✅ Can create visuals like "Top 10 Drifted Features"

**Option B: Modify ETL Pipeline**
Modify `lambda_drift_monitor.py` to write a separate table:

```python
# In lambda_drift_monitor.py, after computing drift scores:
for feature, metrics in feature_drift_scores.items():
    write_to_table(
        table='feature_drift_scores',
        data={
            'monitoring_run_id': run_id,
            'feature_name': feature,
            'drift_score': metrics['drift_score'],
            'drift_detected': metrics['drift_detected'],
            'psi_score': metrics.get('psi', 0)
        }
    )
```

**Benefits:**
- ✅ No post-processing needed
- ✅ Optimized for QuickSight querying
- ✅ Can add pre-computed aggregations

**Option C: QuickSight Calculated Fields**
For small feature sets (< 30 features), create calculated fields:

```
Transaction Hour Drift = parseJson({per_feature_drift_scores}, '$.transaction_hour.drift_score')
Transaction Amount Drift = parseJson({per_feature_drift_scores}, '$.transaction_amount.drift_score')
...
```

**Pros:** Works with existing data
**Cons:** Tedious, not scalable for many features

## Files Modified

1. **`src/config/config.yaml`**
   - Added `feature_drift_dataset_id` and `feature_drift_dataset_name`

2. **`src/config/config.py`**
   - Added `QUICKSIGHT_FEATURE_DRIFT_DATASET_ID`
   - Added `QUICKSIGHT_FEATURE_DRIFT_DATASET_NAME`

3. **`notebooks/3_governance_dashboard.ipynb`**
   - Cell 2: Added imports for new config variables
   - Cell 15-16: Feature drift dataset creation (Custom SQL)
   - Cell 21-22: Feature drift visuals definition
   - Cell 24: Updated analysis to include Sheet 3
   - Cell 26: Updated dashboard to include Sheet 3
   - Cell 30: Updated cleanup to delete new dataset

4. **`notebooks/3_governance_dashboard_feature_drift.md`**
   - Design notes and implementation guide

5. **`notebooks/GOVERNANCE_DASHBOARD_SUMMARY.md`**
   - This file - comprehensive documentation

## Running the Notebook

```bash
# 1. Ensure data exists in Athena tables
python3 -m src.drift_monitoring.create_monitoring_table

# 2. Run the governance dashboard notebook
jupyter notebook notebooks/3_governance_dashboard.ipynb

# 3. Execute all cells sequentially

# 4. Open the dashboard
# URL printed in output:
# https://us-east-1.quicksight.aws.amazon.com/sn/dashboards/fraud-governance-dashboard
```

## Verification

After running the notebook, verify:

1. **Datasets Created:**
   ```bash
   aws quicksight describe-data-set \
     --aws-account-id <account-id> \
     --data-set-id fraud-governance-feature-drift-dataset
   ```

2. **Sheet 3 Exists:**
   - Open dashboard in QuickSight
   - Should see 3 tabs at bottom
   - Click "Feature Drift Analysis" tab

3. **Visuals Render:**
   - All 4 visuals should load without errors
   - If "No data", check that:
     - monitoring_responses table has data
     - inference_responses has matching model_version values

## Troubleshooting

**"No data in visuals"**
- Run drift monitoring Lambda at least once
- Ensure both tables have data with matching `model_version`
- Check date ranges (default shows last 30 days)

**"Custom SQL error"**
- Verify table names in config match actual Athena tables
- Check Lake Formation permissions (see Cell 10)
- Review CloudWatch Logs for QuickSight errors

**"DataSetArn not found"**
- Run cells 15-16 before cells 24-26
- Ensure dataset creation succeeded (check for ✓ checkmark)
- Restart notebook kernel if variables not defined

## Next Steps

1. **Enable Real-Time Refresh**
   - Switch `ImportMode` to `SPICE` for faster queries
   - Set up scheduled refresh (hourly or daily)

2. **Add Alerts**
   - Create QuickSight threshold alerts
   - Trigger SNS when drift > X% for any model

3. **Feature-Level Detail**
   - Implement Option A or B (see Recommended Enhancements)
   - Add Sheet 4 with per-feature drill-down

4. **Custom Metrics**
   - Add business metrics (e.g., transaction amount trends)
   - Join with external data (e.g., weather, holidays)

## Support

For issues:
- Check `3_governance_dashboard_feature_drift.md` for design details
- Review AWS QuickSight documentation
- Check CloudWatch Logs: `/aws/quicksight/`
- GitHub issues: [repository URL]
