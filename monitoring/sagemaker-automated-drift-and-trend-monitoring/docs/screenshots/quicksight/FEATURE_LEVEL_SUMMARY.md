# Feature-Level Drift Analysis - Implementation Summary

## ✅ What's Already Done

### 1. Athena View Created
**View Name**: `fraud_detection.feature_drift_detail`

This view unpacks the JSON `per_feature_drift_scores` column into individual rows:
- ✅ Created and tested
- ✅ Contains data from 3 monitoring runs
- ✅ Tracks 29 features per run
- ✅ Provides drift_score, drift_severity, drift_detected for each feature

**Sample Data Available**:
```
Top 5 Drifting Features:
1. credit_limit: 74.47 avg drift (CRITICAL!)
2. merchant_category_code: 28.25 avg drift
3. account_age_days: 5.80 avg drift
4. max_transaction_amount_30days: 4.17 avg drift
5. time_since_last_transaction_min: 1.41 avg drift
```

### 2. Documentation Created
- ✅ `feature_level_drift_analysis.md` - Full guide with use cases
- ✅ `add_feature_level_visuals.py` - Code to add to notebook
- ✅ SQL queries and visualization recommendations

## 📊 What You Can Build

### Sheet 4: Feature Drift Detail (6 Visuals)

**Visual 17: Feature Drift Timeline**
- Line chart showing selected features' drift over time
- Compare 3-5 features side-by-side
- Identify temporal patterns

**Visual 18: Top 15 Drifting Features**
- Horizontal bar chart
- Shows average drift score per feature
- Quickly identify problematic features

**Visual 19: Drift Severity Distribution**
- Stacked bar chart by drift severity (Low/Moderate/Significant)
- Shows how often each feature drifts at each severity level
- Helps prioritize which features to investigate

**Visual 20: Feature Drift Detail Table**
- Detailed table with all feature drift data
- Sortable and filterable
- Export capability for analysis

**Visual 21: Highest Drifting Feature KPI**
- Shows the worst feature currently
- Real-time alert indicator

**Visual 22: Feature Drift Heatmap**
- Pivot table: features × time
- Color-coded by drift score
- Visual pattern recognition

## 🎯 Use Cases Supported

### 1. "Which features are causing my model drift?"
**Answer**: Sheet 4, Visual 18
- Shows top drifting features immediately
- Your current issue: `credit_limit` has 74x drift!

### 2. "Has feature X stabilized after my data fix?"
**Answer**: Sheet 4, Visual 17
- Filter to specific feature
- See drift trend over time
- Confirm it drops below 0.1 (no drift threshold)

### 3. "Compare features across model versions"
**Answer**: Sheet 4, Visual 17 + filter by model_version
- See if new model handles features better
- Identify if retraining helped specific features

### 4. "Set up proactive alerts"
**Answer**: Sheet 4, Visual 20 + QuickSight Alerts
- Alert when critical features exceed threshold
- Catch issues before they impact production

## 🚀 Implementation Steps

### Step 1: Add Dataset (5 minutes)
1. Open `3_governance_dashboard.ipynb`
2. Insert new cell after Cell 17 (feature drift dataset)
3. Paste code from `add_feature_level_visuals.py` → CELL 1
4. Run cell → creates `FEATURE_LEVEL_DATASET_ARN`

### Step 2: Add Visuals (5 minutes)
1. Insert new cell after your last visual cell
2. Paste code from `add_feature_level_visuals.py` → CELL 2
3. Run cell → defines `FEATURE_LEVEL_VISUALS`

### Step 3: Update Analysis & Dashboard (2 minutes)
1. Find your analysis creation cell
2. Update `analysis_definition` to include Sheet 4
3. Find your dashboard creation cell
4. Update `dashboard_definition` to include Sheet 4
5. Run both cells

**Reference**: See CELL 3 in `add_feature_level_visuals.py` for exact code

### Step 4: Verify (2 minutes)
1. Open dashboard URL
2. Click "Feature Drift Detail" tab
3. Should see 6 visuals with data
4. Test filtering by feature_name

## 📈 Expected Results

After implementation, your dashboard will have:
- **4 sheets** (was 3)
- **22 visuals** (was 16)
- **Feature-level granularity** for drift analysis

### Data Volume
With 3 monitoring runs × 29 features = 87 rows in the view
- Fast queries (< 1 second)
- Plenty of data for trend analysis
- Will grow linearly with monitoring runs

## ⚠️ Immediate Action Recommended

Your data shows **critical drift** in `credit_limit`:
- Drift score: 74.47 (threshold is 0.1)
- This is 744x the "no drift" threshold!
- All 3 runs show significant drift

**Investigate**:
1. Check if credit_limit data format changed
2. Verify feature engineering pipeline
3. Check for data quality issues in source
4. May need to retrain model with new distribution

## 🔍 Query Examples

### Check Specific Feature History
```python
# Run this in notebook after dataset creation
query = f"""
SELECT
    monitoring_timestamp,
    drift_score,
    drift_severity,
    model_version
FROM fraud_detection.feature_drift_detail
WHERE feature_name = 'credit_limit'
ORDER BY monitoring_timestamp
"""
# Execute via Athena or QuickSight
```

### Find Consistently Drifting Features
```sql
SELECT
    feature_name,
    COUNT(*) as total_runs,
    SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) as drifted_runs,
    AVG(drift_score) as avg_drift
FROM fraud_detection.feature_drift_detail
GROUP BY feature_name
HAVING SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) >= 2  -- Drifted in 2+ runs
ORDER BY avg_drift DESC
```

## 📚 Files Created

1. **`docs/guides/claude/feature_level_drift_analysis.md`**
   - Complete guide with SQL queries
   - Use cases and recommendations
   - Data structure documentation

2. **`docs/guides/claude/add_feature_level_visuals.py`**
   - Copy-paste code for notebook
   - All 6 visual definitions
   - Dataset configuration

3. **`docs/screenshots/quicksight/FEATURE_LEVEL_SUMMARY.md`** (this file)
   - Implementation checklist
   - Quick reference

## ✅ Checklist

- [x] Athena view created (`feature_drift_detail`)
- [x] View tested and contains data
- [x] Documentation written
- [x] Visual definitions created
- [ ] Dataset added to notebook
- [ ] Visuals added to notebook
- [ ] Analysis updated with Sheet 4
- [ ] Dashboard updated with Sheet 4
- [ ] Dashboard published and verified
- [ ] Investigated credit_limit drift issue

## 🎓 Key Insights from Your Data

1. **credit_limit** drifts massively (74x) - **ACTION REQUIRED**
2. **merchant_category_code** shows consistent high drift (28x)
3. Most features drift at a "significant" level in all runs
4. This suggests a fundamental data distribution shift
5. Model retraining may be necessary if distributions have permanently changed

## Next Steps

1. **Add Sheet 4** to dashboard (15 minutes total)
2. **Investigate credit_limit** drift (high priority)
3. **Set up alerts** for top 5 drifting features
4. **Review model performance** - high drift usually correlates with degraded accuracy
5. **Consider retraining** if drift is permanent

## Support

- Full documentation: `docs/guides/claude/feature_level_drift_analysis.md`
- Code to add: `docs/guides/claude/add_feature_level_visuals.py`
- Athena view: `fraud_detection.feature_drift_detail` (already created)
