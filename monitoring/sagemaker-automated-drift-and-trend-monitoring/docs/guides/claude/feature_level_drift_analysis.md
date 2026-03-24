# Feature-Level Drift Analysis

## Overview

The feature-level drift analysis shows how individual features drift over time across multiple monitoring runs. This helps identify which specific features are causing data drift.

## Data Source

**Athena View**: `fraud_detection.feature_drift_detail`

This view unpacks the JSON `per_feature_drift_scores` column from `monitoring_responses` into individual rows:

```sql
CREATE OR REPLACE VIEW fraud_detection.feature_drift_detail AS
SELECT
    monitoring_run_id,
    monitoring_timestamp,
    model_version,
    endpoint_name,
    data_drift_detected,
    drifted_columns_count,
    drifted_columns_share,
    baseline_roc_auc,
    current_roc_auc,
    feature_name,                    -- Unpacked from JSON
    drift_score,                     -- Unpacked from JSON
    CASE
        WHEN drift_score > 0.25 THEN 'Significant'
        WHEN drift_score > 0.1 THEN 'Moderate'
        ELSE 'Low'
    END as drift_severity,           -- Computed severity
    CASE WHEN drift_score > 0.1 THEN true ELSE false END as drift_detected
FROM fraud_detection.monitoring_responses
CROSS JOIN UNNEST(
    CAST(json_parse(per_feature_drift_scores) AS MAP(VARCHAR, DOUBLE))
) AS t(feature_name, drift_score)
WHERE per_feature_drift_scores IS NOT NULL
```

## Data Structure

Each row represents one feature in one monitoring run:

| Column | Type | Description |
|--------|------|-------------|
| monitoring_run_id | STRING | Unique run identifier |
| monitoring_timestamp | DATETIME | When monitoring ran |
| model_version | STRING | Model version monitored |
| feature_name | STRING | Name of the feature (e.g., "transaction_amount") |
| drift_score | DECIMAL | PSI drift score for this feature |
| drift_severity | STRING | 'Low', 'Moderate', or 'Significant' |
| drift_detected | BOOLEAN | True if drift_score > 0.1 |

## Example Queries

### Top 10 Most Drifting Features
```sql
SELECT
    feature_name,
    COUNT(*) as run_count,
    AVG(drift_score) as avg_drift,
    MAX(drift_score) as max_drift,
    SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) as times_drifted
FROM fraud_detection.feature_drift_detail
GROUP BY feature_name
ORDER BY avg_drift DESC
LIMIT 10
```

### Feature Drift Timeline
```sql
SELECT
    monitoring_timestamp,
    feature_name,
    drift_score,
    drift_severity
FROM fraud_detection.feature_drift_detail
WHERE feature_name IN ('transaction_amount', 'customer_age', 'credit_limit')
ORDER BY monitoring_timestamp, feature_name
```

### Features with Consistent Drift
```sql
SELECT
    feature_name,
    COUNT(*) as total_runs,
    SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) as drifted_runs,
    CAST(SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) as drift_rate
FROM fraud_detection.feature_drift_detail
GROUP BY feature_name
HAVING drift_rate > 0.5  -- Drifts in >50% of runs
ORDER BY drift_rate DESC
```

## Recommended Visuals for QuickSight

### Sheet 4: Feature-Level Drift Detail

**Visual 1: Feature Drift Timeline (Line Chart)**
- X-axis: monitoring_timestamp
- Y-axis: drift_score
- Color: feature_name (use filter to select 3-5 features)
- Shows how specific features drift over time

**Visual 2: Top Drifting Features (Horizontal Bar Chart)**
- Y-axis: feature_name
- X-axis: AVG(drift_score)
- Sort: Descending by avg drift
- Limit: Top 15
- Shows which features consistently drift the most

**Visual 3: Drift Heatmap (Pivot Table)**
- Rows: feature_name
- Columns: DATE_TRUNC('day', monitoring_timestamp)
- Values: drift_score (color gradient)
- Shows drift patterns across all features and time

**Visual 4: Feature Drift Distribution (Histogram)**
- X-axis: drift_score (binned)
- Y-axis: COUNT of features
- Shows distribution of drift severity across all features

**Visual 5: Drift Severity by Feature (Stacked Bar)**
- Y-axis: feature_name (Top 15 by total drift)
- X-axis: COUNT(*)
- Stack by: drift_severity
- Shows how often each feature is Low/Moderate/Significant

**Visual 6: Feature Drift Detail Table**
- Columns: feature_name, monitoring_timestamp, drift_score, drift_severity, model_version
- Filters: feature_name, drift_severity, date range
- Sortable and searchable

## QuickSight Dataset Configuration

**Dataset ID**: `fraud-governance-feature-level-dataset`
**Dataset Name**: `Fraud Governance - Feature Level Drift`

**Physical Table Map**:
```python
{
    'feature-level-view': {
        'RelationalTable': {
            'DataSourceArn': DATASOURCE_ARN,
            'Catalog': 'AwsDataCatalog',
            'Schema': 'fraud_detection',
            'Name': 'feature_drift_detail',  # The Athena view
            'InputColumns': [
                {'Name': 'monitoring_run_id', 'Type': 'STRING'},
                {'Name': 'monitoring_timestamp', 'Type': 'DATETIME'},
                {'Name': 'model_version', 'Type': 'STRING'},
                {'Name': 'endpoint_name', 'Type': 'STRING'},
                {'Name': 'feature_name', 'Type': 'STRING'},
                {'Name': 'drift_score', 'Type': 'DECIMAL'},
                {'Name': 'drift_severity', 'Type': 'STRING'},
                {'Name': 'drift_detected', 'Type': 'BIT'},
                {'Name': 'drifted_columns_count', 'Type': 'INTEGER'},
                {'Name': 'drifted_columns_share', 'Type': 'DECIMAL'},
                {'Name': 'baseline_roc_auc', 'Type': 'DECIMAL'},
                {'Name': 'current_roc_auc', 'Type': 'DECIMAL'},
            ],
        }
    }
}
```

## Use Cases

### 1. Identify Root Cause of Drift
**Scenario**: Aggregate drift is detected, need to find which features are causing it

**How to use**:
- Sheet 4, Visual 2 (Top Drifting Features)
- Look at the top 5 features with highest avg drift
- Check their business logic - are they derived features? External data?
- Investigate data pipeline for these specific features

### 2. Track Feature Stability Over Time
**Scenario**: After fixing a data issue, verify the feature stabilizes

**How to use**:
- Sheet 4, Visual 1 (Feature Drift Timeline)
- Filter to the specific feature (e.g., "credit_limit")
- Check if drift score decreases after the fix date
- Confirm drift_severity changes from 'Significant' to 'Low'

### 3. Compare Feature Drift Across Models
**Scenario**: Multiple model versions deployed, want to see which handles features better

**How to use**:
- Sheet 4, Visual 1 (Timeline)
- Add filter by model_version
- Compare same feature across versions
- Identify if new model version has better feature stability

### 4. Proactive Monitoring Alerts
**Scenario**: Set up alerts for specific high-value features

**How to use**:
- Sheet 4, Visual 6 (Detail Table)
- Filter to critical features (e.g., transaction_amount, customer_age)
- Sort by drift_score descending
- Set QuickSight threshold alert when drift_score > 0.25

## Sample Data

From your current monitoring runs:

**Top 5 Drifting Features:**
1. **credit_limit**: 74.47 avg drift (VERY HIGH - investigate!)
2. **merchant_category_code**: 28.25 avg drift (HIGH)
3. **account_age_days**: 5.80 avg drift (HIGH)
4. **max_transaction_amount_30days**: 4.17 avg drift (HIGH)
5. **time_since_last_transaction_min**: 1.41 avg drift (MODERATE)

## Implementation Steps

1. **Create Athena View** (Already done ✅)
   ```sql
   CREATE OR REPLACE VIEW fraud_detection.feature_drift_detail AS ...
   ```

2. **Add QuickSight Dataset**
   - Run new cell in notebook to create dataset
   - Dataset will reference the Athena view

3. **Create Visuals**
   - Add Sheet 4 to the dashboard
   - Create 6 visuals as described above
   - Add filters for feature selection

4. **Configure Refresh**
   - Set to DIRECT_QUERY mode (queries Athena in real-time)
   - Or use SPICE with scheduled refresh

## Benefits

✅ **Granular Insights**: See which specific features are problematic
✅ **Historical Tracking**: Understand drift patterns over time
✅ **Root Cause Analysis**: Quickly identify data quality issues
✅ **Proactive Monitoring**: Alert on critical feature drift before it impacts models
✅ **Model Comparison**: Compare feature stability across model versions

## Next Steps

1. Add the feature-level dataset to your notebook
2. Create Sheet 4 with the recommended visuals
3. Set up alerts for critical features (credit_limit, merchant_category_code)
4. Investigate why credit_limit has 74x drift (this is extremely high!)
5. Consider retraining model if feature distributions have fundamentally changed
